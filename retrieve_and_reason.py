import os
import torch
import fitz
import base64
from google import genai
from PIL import Image
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\huggingface"
os.environ["TORCH_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\torch"

# API Configuration Fallback to Modern Gemini SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def extract_pdf_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def main():
    print("--- Digital Mechanic M-RAG Retrieval + Gemini Started ---")
    
    qdrant_path = "E:\\SACHA\\Multimodal Rag\\qdrant_data"
    qdrant = QdrantClient(path=qdrant_path)

    user_query = input("\nEnter your mechanical problem: ")
    if not user_query.strip():
        user_query = "What is the exact torque specification for the spark plugs?"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    
    print("\n[1] Encoding your query using Jina-CLIP into 768-D Space...")
    model_name = "jinaai/jina-clip-v1"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    inputs = processor(text=[user_query], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_vector = text_features[0].cpu().float().numpy().tolist()

    del model
    del processor
    del inputs

    print("\n[2] Performing Hybrid Semantic Search across both Manuals & YouTube Videos...")
    
    manual_results = []
    if qdrant.collection_exists("ford_manuals"):
        manual_results = qdrant.query_points(collection_name="ford_manuals", query=query_vector, limit=2).points
    
    video_results = []
    if qdrant.collection_exists("mechanic_videos"):
        video_results = qdrant.query_points(collection_name="mechanic_videos", query=query_vector, limit=2).points
    
    # Prepare Native Gemini PIL Sequence Payload
    prompt_elements = [
        f"USER QUERY: {user_query}\n\n",
        "You are an expert 'Digital Mechanic'. Using the attached Visual Context (PDF Manual diagrams and YouTube Video keyframes) plus the provided diagnostic video transcript blocks, structure precise repair instructions.",
        "Provide steps, torque specifications if applicable, and reference the visual context naturally. Do not hallucinate capabilities outside the attachments.",
        "\n--- ATTACHED VISUAL AND DATA CONTEXT ---"
    ]
    
    print("\n[3] Extracting Matched Hardware Contexts...")
    for hit in manual_results:
        pdf_file = hit.payload.get("doc_name")
        page_no = int(hit.payload.get("page_num"))
        print(f"    - Found Schema Manual Match: {pdf_file} Page {page_no} (Score: {hit.score:.4f})")
        
        img = extract_pdf_image(pdf_file, page_no)
        prompt_elements.append(f"\n[DOCUMENT SOURCE: {pdf_file} - Page {page_no}]")
        prompt_elements.append(img)
        
    for hit in video_results:
        vid = hit.payload.get("video_id")
        ts = hit.payload.get("timestamp")
        frame_path = hit.payload.get("frame_path")
        transcript = hit.payload.get("transcript", "")
        print(f"    - Found YouTube Frame Match: Video {vid} @ {ts:.2f}s (Score: {hit.score:.4f})")
        
        if os.path.exists(frame_path):
            img = Image.open(frame_path)
            prompt_elements.append(f"\n[YOUTUBE SOURCE: Video {vid} Keyframe at {ts:.2f} seconds]")
            prompt_elements.append(img)
            if transcript:
                prompt_elements.append(f"Transcript Context from {vid}: {transcript[:500]}...")

    print("\n[4] Engaging Modern Gemini Network Core Engine...")
    try:
        response = gemini_client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt_elements
        )
        print("\n================ DIGITAL MECHANIC DIAGNOSIS ================\n")
        print(response.text)
        print("\n============================================================")
    except Exception as e:
        print(f"CRITICAL Gemini API Runtime Error: {e}")

if __name__ == "__main__":
    main()
