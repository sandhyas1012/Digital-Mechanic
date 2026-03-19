import os
import cv2
import yt_dlp
import uuid
import torch
import shutil
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoModel, AutoProcessor
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\huggingface"
os.environ["TORCH_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\torch"

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
FRAMES_DIR = "E:\\SACHA\\Multimodal Rag\\video_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

def search_youtube_videos(query, max_results=3):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=query, part='snippet', type='video', maxResults=max_results
    )
    response = request.execute()
    return [item['id']['videoId'] for item in response.get('items', [])]

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t['text'] for t in transcript])
        return full_text
    except Exception as e:
        print(f"Transcript error for {video_id}: {e}")
        return ""

def download_video(video_id, download_path):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best[ext=mp4]',
        'outtmpl': f'{download_path}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def extract_keyframes(video_path, video_id, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_info = []
    if total_frames <= 0:
        return frames_info
        
    step = max(1, total_frames // num_frames)
    
    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(im_rgb)
            frame_filename = os.path.join(FRAMES_DIR, f"{video_id}_frame_{i}.jpg")
            pil_img.save(frame_filename)
            
            timestamp = frame_idx / fps if fps > 0 else 0
            frames_info.append({
                "path": frame_filename,
                "timestamp": timestamp,
                "pil_img": pil_img
            })
            
    cap.release()
    return frames_info

def main():
    print("--- Digital Mechanic: YouTube Enrichment Module ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    
    print(f"Loading Jina-CLIP on {device}...")
    model_name = "jinaai/jina-clip-v1"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    qdrant_path = "E:\\SACHA\\Multimodal Rag\\qdrant_data"
    client = QdrantClient(path=qdrant_path)
    collection_name = "mechanic_videos"
    
    if client.collection_exists(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)
    
    print(f"Initializing Qdrant Keyframe Collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
        
    query = "Ford Focus 2005 spark plug replacement repair"
    print(f"\nSearching YouTube API for: '{query}'")
    video_ids = search_youtube_videos(query, max_results=2)
    print(f"Found diagnostic videos: {video_ids}")
    
    for vid in video_ids:
        print(f"\n[Processing Video: {vid}]")
        transcript = get_transcript(vid)
        
        temp_dir = "E:\\SACHA\\Multimodal Rag\\temp_downloads"
        os.makedirs(temp_dir, exist_ok=True)
        try:
            print("  -> Downloading high-compression video stream via yt-dlp...")
            video_path = download_video(vid, temp_dir)
            
            print("  -> Slicing keyframes natively using OpenCV...")
            frames = extract_keyframes(video_path, vid, num_frames=5)
            
            for f in frames:
                pil_img = f["pil_img"]
                # Embed frame purely visually
                inputs = processor(images=[pil_img], return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vector = image_features[0].cpu().float().numpy().tolist()
                
                point_id = str(uuid.uuid4())
                points = [
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "type": "youtube_frame",
                            "video_id": vid,
                            "timestamp": f["timestamp"],
                            "frame_path": f["path"],
                            "transcript": transcript[:1500] # Store first part of transcript contextually
                        }
                    )
                ]
                client.upsert(collection_name=collection_name, wait=True, points=points)
            print(f"  -> Encoded 5 frames to Qdrant successfully.")
                
        except Exception as e:
            print(f"Error processing {vid}: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    print("\nYouTube enrichment securely finished. Keyframes are embedded in local storage!")

if __name__ == "__main__":
    main()
