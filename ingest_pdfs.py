import os
import torch
import uuid
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoModel, AutoProcessor

# CRITICAL CONSTRAINT: Set cache directories to E: drive
os.environ["HF_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\huggingface"
os.environ["HF_DATASETS_CACHE"] = "E:\\SACHA\\Multimodal Rag\\.cache\\huggingface\\datasets"
os.environ["TORCH_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\torch"

os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)


def get_page_image(pdf_path, page_num):
    """
    Renders an ultra-low memory strict visual image of a single PDF page.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    # Using 150 DPI provides sufficient clarity for the text without blowing out RAM/VRAM
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def main():
    print("Initializing Lightweight Multimodal RAG Ingestion Pipeline...")
    
    # 1. Setup device and Jina-CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Prevent PyTorch from spawning excess threads that spike RAM
    torch.set_num_threads(1)

    model_name = "jinaai/jina-clip-v1"
    print(f"Loading Jina-CLIP model: {model_name}")
    
    # Jina-CLIP requires trust_remote_code=True
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    # 2. Setup Qdrant
    qdrant_path = "E:\\SACHA\\Multimodal Rag\\qdrant_data"
    print(f"Connecting to Qdrant Local at {qdrant_path}...")
    client = QdrantClient(path=qdrant_path)
    collection_name = "ford_manuals"

    if client.collection_exists(collection_name=collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    print(f"Creating collection: {collection_name}")
    # Jina-CLIP embeddings are 768-dimensional. We use COSINE distance for normalized embeddings.
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    pdf_files = [
        "ford 2005 Focus Workshop manual.pdf",
        "Electrical wiring.pdf",
        "Ford 8700, 9700 Repair Manual.pdf"
    ]

    # 3. Ingest Documents
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"Warning: Could not find {pdf_file}, skipping.")
            continue
            
        try:
            doc_info = fitz.open(pdf_file)
            total_pages = len(doc_info)
            doc_info.close()
        except Exception as e:
            print(f"Could not read page count for {pdf_file}: {e}")
            continue
            
        print(f"\nProcessing {pdf_file} [{total_pages} total pages] natively with PyMuPDF...")
        
        # Iterate through sequentially to guarantee zero bad_alloc crashes
        for page_idx in range(total_pages):
            page_no = page_idx + 1
            print(f"  Embedding page {page_no}/{total_pages} of {pdf_file}...")
            
            # Step A: Minimal RAM image render
            pil_img = get_page_image(pdf_file, page_idx)
            
            # Step B: Model embedding block
            inputs = processor(images=[pil_img], return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize features for Cosine Similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            # Grab the 768-dim vector for this page
            vector = image_features[0].cpu().float().numpy().tolist()
            
            # Insert visually-embedded vector strictly tied to the page number
            point_id = str(uuid.uuid4())
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "doc_name": pdf_file,
                        "page_num": int(page_no)
                    }
                )
            ]
                
            client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            
            # Force memory cleanup
            del pil_img
            del inputs
            del image_features
            
    print("\nFinished ingestion completely natively into Qdrant Local Storage using Jina-CLIP without Bloat!")

if __name__ == "__main__":
    main()
