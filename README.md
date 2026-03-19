# Digital Mechanic: Multimodal RAG Assistant

An advanced **Multimodal Retrieval-Augmented Generation (M-RAG)** AI application designed to act as a "Digital Mechanic". It visually analyzes damage photos, reads mechanical repair manuals, and scrapes YouTube tutorials to provide specific, step-by-step repair guides.

## Features
- **Visual Diagnostics:** Upload a photo of a damaged car part; the AI will analyze it visually.
- **Ford Workshop Integration:** Embeds and searches actual Ford Focus & Tractor repair manuals (PDFs) and electrical wiring diagrams.
- **YouTube Enrichment:** Dynamically fetches related YouTube repair keyframes and transcripts using `yt-dlp` and `OpenCV`.
- **True Multimodal Embedding:** Uses `jina-clip-v1` to encode both Text and Visual Images into the exact same 768-dimensional vector space stored in local `Qdrant`.
- **Reasoning Engine:** Uses Google `Gemini Flash` to reason simultaneously across the user query, uploaded damage photo, retrieved PDF diagrams, and YouTube frames to generate step-by-step verified instructions.

## Tech Stack
- **UI:** Gradio
- **Embedding Agent:** Jina-CLIP (HuggingFace, PyTorch)
- **Vector Database:** Qdrant (Local Storage)
- **Generative AI:** Google Gemini GenAI SDK
- **Data Ingestion:** PyMuPDF (`fitz`), yt-dlp, OpenCV, YouTube Transcript API

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Multimodal Rag"
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Make sure you have your keys ready in your environment or placed in the scripts:
   - `GEMINI_API_KEY`
   - `YOUTUBE_API_KEY`

5. **Add your PDFs:**
   Place your PDF manuals (like `ford 2005 Focus Workshop manual.pdf`) into the root directory.

## How to Run

For convenience, you can just run the initialization batch script which will orchestrate the data ingestion and start the app:
```bash
.\run_all.bat
```

Alternatively, you can run the modules individually:

1. **Ingest PDFs:**
   ```bash
   python ingest_pdfs.py
   ```
2. **Ingest YouTube Videos:**
   ```bash
   python youtube_scraper.py
   ```
3. **Start the Web UI:**
   ```bash
   python app.py
   ```
   *Then open `http://127.0.0.1:7860` in your browser.*

## Architecture Overview
1. **Ingestion Pipelines:** Convert PDF pages and YouTube video frames strictly into visual RGB images and embed them directly into Qdrant using `jina-clip-v1`.
2. **RAG Retrieval:** Jina-CLIP encodes the user's text or image query into a vector and pulls the top *Multi-modal* nodes (visual schemas, diagrams, keyframes).
3. **Synthesis:** Gemini ingests all the raw images and retrieved data in a single massive prompt sequence, producing a conversational repair guide without hallucinations.
