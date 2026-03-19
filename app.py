import os
import re
import torch
import fitz
import urllib.parse
from PIL import Image
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoProcessor
from google import genai
from googleapiclient.discovery import build
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\huggingface"
os.environ["TORCH_HOME"] = "E:\\SACHA\\Multimodal Rag\\.cache\\torch"

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
YOUTUBE_API_KEY  = os.getenv("YOUTUBE_API_KEY", "")

if not GEMINI_API_KEY or not YOUTUBE_API_KEY:
    print("WARNING: Please set GEMINI_API_KEY and YOUTUBE_API_KEY environment variables.")
QDRANT_PATH      = "E:\\SACHA\\Multimodal Rag\\qdrant_data"
MODEL_NAME       = "jinaai/jina-clip-v1"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

_processor = _clip_model = _qdrant = _device = None

def lazy_init():
    global _processor, _clip_model, _qdrant, _device
    if _clip_model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _clip_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(_device)
    _clip_model.eval()
    _qdrant = QdrantClient(path=QDRANT_PATH)


def extract_pdf_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=130)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def encode_image(pil_img):
    inputs = _processor(images=[pil_img], return_tensors="pt").to(_device)
    with torch.no_grad():
        f = _clip_model.get_image_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().float().numpy().tolist()


def encode_text(text):
    inputs = _processor(text=[text], return_tensors="pt").to(_device)
    with torch.no_grad():
        f = _clip_model.get_text_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().float().numpy().tolist()


def youtube_suggestions(query, max_results=3):
    """Return markdown links to the top YouTube tutorials for this query."""
    try:
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        resp = yt.search().list(q=query + " repair tutorial", part="snippet",
                                type="video", maxResults=max_results).execute()
        lines = []
        for item in resp.get("items", []):
            vid_id = item["id"]["videoId"]
            title  = item["snippet"]["title"]
            lines.append(f"- 🎬 [{title}](https://www.youtube.com/watch?v={vid_id})")
        return "\n".join(lines) if lines else "_No videos found._"
    except Exception as e:
        return f"_YouTube lookup failed: {e}_"


def maps_link(fault_keyword):
    """Return a Google Maps search link for nearby repair shops."""
    query = urllib.parse.quote(f"{fault_keyword} car repair shop near me")
    return f"https://www.google.com/maps/search/{query}"


def diagnose(user_image, user_question):
    lazy_init()

    if user_image is None and not user_question.strip():
        return "⚠️ Please upload a photo OR type a question (or both).", ""

    status_lines = []

    # ── 1. Encode the query ──────────────────────────────────────────────────
    if user_image is not None:
        status_lines.append("🔍 Using your uploaded photo as the visual query…")
        pil_upload = Image.fromarray(user_image).convert("RGB")
        query_vector = encode_image(pil_upload)
        status_lines.append("✅ Photo encoded with Jina‑CLIP.")
    else:
        status_lines.append("🔍 Using your text question as the query…")
        pil_upload = None
        query_vector = encode_text(user_question)
        status_lines.append("✅ Text encoded with Jina‑CLIP.")

    # ── 2. Retrieve from Qdrant ──────────────────────────────────────────────
    manual_hits, video_hits = [], []
    if _qdrant.collection_exists("ford_manuals"):
        manual_hits = _qdrant.query_points(collection_name="ford_manuals",
                                           query=query_vector, limit=2).points
    if _qdrant.collection_exists("mechanic_videos"):
        video_hits  = _qdrant.query_points(collection_name="mechanic_videos",
                                           query=query_vector, limit=2).points

    status_lines.append(f"📚 Found {len(manual_hits)} manual page(s) + {len(video_hits)} YouTube frame(s).")

    # ── 3. Build Gemini prompt ───────────────────────────────────────────────
    prompt_parts = [
        "You are an expert Digital Mechanic specialising in Ford vehicles "
        "(2005 Focus & Ford 8700/9700 Tractor). "
        "Using ALL attached images (user damage photo + workshop manual diagrams + "
        "YouTube repair keyframes) provide a clear, friendly, step-by-step repair answer. "
        "Format your response neatly using Markdown (headers, bullet points, and bold text) for easy reading. "
        "Use numbered steps, short paragraphs, and mention torque specs in parentheses where relevant. "
        "Do NOT hallucinate parts or procedures not visible in the evidence."
    ]

    if pil_upload is not None:
        prompt_parts.append("\n[USER DAMAGE PHOTO — identify the fault:]")
        prompt_parts.append(pil_upload)

    if user_question.strip():
        prompt_parts.append(f"\nUser asks: {user_question}")

    for hit in manual_hits:
        pdf_file = hit.payload.get("doc_name")
        page_no  = int(hit.payload.get("page_num"))
        status_lines.append(f"   📄 {pdf_file} — Page {page_no} (score {hit.score:.3f})")
        try:
            img = extract_pdf_image(pdf_file, page_no)
            prompt_parts += [f"\n[Workshop Manual: {pdf_file}, Page {page_no}]", img]
        except Exception as e:
            status_lines.append(f"   ⚠️ Page load error: {e}")

    for hit in video_hits:
        vid       = hit.payload.get("video_id")
        ts        = hit.payload.get("timestamp", 0)
        fp        = hit.payload.get("frame_path", "")
        transcript= hit.payload.get("transcript", "")
        status_lines.append(f"   🎬 YouTube {vid} @ {ts:.1f}s (score {hit.score:.3f})")
        if fp and os.path.exists(fp):
            try:
                frame_img = Image.open(fp)
                prompt_parts += [f"\n[YouTube {vid} @ {ts:.1f}s]", frame_img]
                if transcript:
                    prompt_parts.append(f"Transcript excerpt: {transcript[:400]}…")
            except Exception as e:
                status_lines.append(f"   ⚠️ Frame load error: {e}")

    # ── 4. Call Gemini ───────────────────────────────────────────────────────
    status_lines.append("\n🤖 Asking Gemini…")
    try:
        resp   = gemini_client.models.generate_content(
            model="gemini-flash-lite-latest", contents=prompt_parts)
        answer = resp.text
    except Exception as e:
        answer = f"❌ Gemini Error: {e}"

    # ── 5. Build "fault keyword" for YouTube & Maps ──────────────────────────
    fault_kw = user_question.strip() if user_question.strip() else "Ford Focus repair"

    # YouTube suggestions via API
    status_lines.append("🎬 Fetching YouTube tutorials…")
    yt_md = youtube_suggestions(fault_kw)

    # Google Maps link
    maps_url = maps_link(fault_kw)
    shops_md = (
        f"[🗺️ Find repair shops near you on Google Maps]({maps_url})"
    )

    # ── 6. Compose final rich markdown answer ────────────────────────────────
    full_answer = f"""{answer}

---

### 🎬 Related YouTube Tutorials
{yt_md}

---

### 🔧 Find a Repair Shop
{shops_md}
"""

    retrieval_log = "\n".join(status_lines)
    return full_answer, retrieval_log


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
#header { text-align:center; padding:24px 0 6px 0; }
#header h1 { font-size:2.3rem; font-weight:800; color:#f97316; }
#header p  { color:#94a3b8; font-size:1rem; margin-top:4px; }
.answer-card { background:#0f172a !important; border-radius:12px; padding:18px; }
.answer-card, .answer-card p, .answer-card li, .answer-card h1, .answer-card h2, .answer-card h3, .answer-card strong { color:#e2e8f0 !important; font-size:0.97rem; line-height:1.7; }
.answer-card a { color:#f97316 !important; font-weight:bold; }
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="orange", secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="Digital Mechanic — Ford M-RAG Assistant",
    css=CSS,
) as demo:

    gr.HTML("""
    <div id="header">
      <h1>🔧 Digital Mechanic</h1>
      <p>Ford Focus 2005 &amp; Ford 8700/9700 Tractor — Multimodal RAG · Jina‑CLIP · Gemini</p>
    </div>""")

    with gr.Row(equal_height=True):
        # ── Left panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            upload_img = gr.Image(
                label="📷 Upload Damage Photo (optional)",
                type="numpy", sources=["upload", "webcam"], height=240,
            )
            question_box = gr.Textbox(
                label="💬 Describe the problem or ask a question",
                placeholder="e.g. What's wrong here? / Torque spec for rear caliper bolts?",
                lines=3,
            )
            submit_btn = gr.Button("🔍 Diagnose", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    [None, "How do I replace the spark plugs on a 2005 Ford Focus?"],
                    [None, "What is the torque spec for the cylinder head bolts?"],
                    [None, "How do I bleed the brakes on a Ford 8700 tractor?"],
                ],
                inputs=[upload_img, question_box],
            )

        # ── Right panel ─────────────────────────────────────────────────────
        with gr.Column(scale=2):
            answer_md = gr.Markdown(
                label="🛠️ Diagnosis & Repair Guide",
                value="*Diagnosis will appear here…*",
                elem_classes=["answer-card"],
            )
            with gr.Accordion("📡 Retrieval Log", open=False):
                log_box = gr.Textbox(lines=6, interactive=False, show_label=False)

    submit_btn.click(
        fn=diagnose,
        inputs=[upload_img, question_box],
        outputs=[answer_md, log_box],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
