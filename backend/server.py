"""
FastAPI Server for Veritas AI — QuillBot-Style Offline AI Detector
Serves:
- REST API for 4-class detection & sentence inspection
- Pre-loaded benchmark samples & 30-sample QuillBot comparison sheet
- Static web dashboard (Plain HTML/JS/CSS)
- Multi-format document upload (PDF, DOCX, TXT)
"""

import os
import json
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.runtime_engine import QuillBotDetectorEngine
from backend.document_parser import parse_uploaded_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMPARISON_SHEET_PATH = os.path.join(DATA_DIR, "quillbot_comparison_sheet.json")

# Initialize FastAPI application
app = FastAPI(
    title="Veritas AI QuillBot-Style Detector Server",
    description="Offline 4-class AI writing detector optimized for low-end hardware",
    version="3.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[QuillBotDetectorEngine] = None


def get_engine() -> QuillBotDetectorEngine:
    global engine
    if engine is None:
        engine = QuillBotDetectorEngine.get_instance(threads=2)
    return engine


class DetectRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.40
    filename: Optional[str] = "Pasted Text"


ARCHETYPE_SAMPLES = {
    "ai_pure": {
        "title": "1. Pure AI-generated (GPT-4o Academic Essay)",
        "expected_class": "AI-generated",
        "text": (
            "In the contemporary era, the rapid proliferation of artificial intelligence technologies has fundamentally "
            "reconstituted the landscape of higher education. To fully appreciate this transformation, one must delve into "
            "the multifaceted tapestry of academic pedagogy. On one hand, automated tutoring systems offer unprecedented "
            "personalization, catering to individual student learning trajectories. On the other hand, the uncritical adoption "
            "of algorithmic tools introduces substantial concerns regarding cognitive atrophy and academic integrity. Ultimately, "
            "fostering an educational ecosystem that harmonizes technological innovation with critical inquiry stands as a "
            "pivotal imperative for educators worldwide."
        )
    },
    "ai_refined_ai": {
        "title": "2. AI-generated & AI-refined (Paraphrased Claude 3.5)",
        "expected_class": "AI-generated & AI-refined",
        "text": (
            "The delicate balance between personal freedom and algorithmic rule highlights a core puzzle in digital law. "
            "Even though online platforms frequently market automated recommendation tools as neutral ways to help users, "
            "looking closely at these systems shows that they actively shape the information we see. Instead of merely "
            "supporting user decisions, software designs fundamentally limit how those choices are formed in the first place. "
            "As a direct result, protecting public discourse requires looking past formal openness and directly questioning "
            "the major power imbalances built into corporate computer systems."
        )
    },
    "human_refined_ai": {
        "title": "3. Human-written & AI-refined (Polished Historical Analysis)",
        "expected_class": "Human-written & AI-refined",
        "text": (
            "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance "
            "between centralized state regulation and private merchant enterprise. Furthermore, the Senate maintained rigorous "
            "oversight of the state galley fleets—the mude—which operated along fixed routes to Constantinople, Alexandria, "
            "and Southampton. Notably, individual patricians frequently invested personal capital in secondary cargo, navigating "
            "volatile price fluctuations and Mediterranean piracy with remarkable institutional flexibility. This hybrid commercial "
            "architecture fostered resilience against geopolitical shocks, particularly following Ottoman expansion into the Aegean."
        )
    },
    "human_pure": {
        "title": "4. Human-written (Venetian Maritime Commerce)",
        "expected_class": "Human-written",
        "text": (
            "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance "
            "between centralized state regulation and private merchant enterprise. The Senate maintained rigorous oversight of "
            "the state galley fleets—the mude—which operated along fixed routes to Constantinople, Alexandria, and Southampton. "
            "However, individual patricians frequently invested personal capital in secondary cargo, navigating volatile price "
            "fluctuations and Mediterranean piracy with remarkable institutional flexibility. This hybrid commercial architecture "
            "fostered resilience against geopolitical shocks, particularly following the Ottoman expansion into the Aegean."
        )
    },
    "human_esl": {
        "title": "5. Human Non-Native / ESL (Learner English Essay)",
        "expected_class": "Human-written",
        "text": (
            "Nowadays, many students choose to study abroad in foreign countries. In my opinion, this experience has many advantages "
            "for young people. Firstly, students can improve their English language skills very quickly because they must speak with "
            "native speakers every day in school and supermarket. Secondly, they can learn how to live independently without their parents' "
            "help, such as cooking food and washing clothes. However, some students feel lonely and miss their hometown food very much. "
            "Therefore, students should prepare their mind carefully before going to study in another country."
        )
    }
}


@app.get("/api/health")
def health_check():
    import psutil
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    return {
        "status": "online",
        "architecture": "Distilled Student ONNX INT8 + Stylometric Meta-Classifier",
        "engine_runtime": "onnxruntime (Zero PyTorch)",
        "device": "cpu",
        "cpu_threads": 2,
        "process_ram_mb": round(rss_mb, 1),
        "target_ram_cap_mb": 1500.0,
        "timestamp": time.time()
    }


@app.get("/api/samples")
def get_samples():
    return ARCHETYPE_SAMPLES


@app.get("/api/comparison-sheet")
def get_comparison_sheet():
    if os.path.exists(COMPARISON_SHEET_PATH):
        with open(COMPARISON_SHEET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/api/detect")
def detect_text(req: DetectRequest):
    det_engine = get_engine()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text.split()) < 5:
        raise HTTPException(status_code=400, detail="Text is too brief. Please enter at least 5 words.")

    try:
        result = det_engine.analyze_text(text, confidence_threshold=req.confidence_threshold or 0.40)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    det_engine = get_engine()
    filename = file.filename or "uploaded_document"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        extracted_text, metadata = parse_uploaded_file(filename, contents)
        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No readable text extracted from document.")

        result = det_engine.analyze_text(extracted_text)
        result["metadata"] = metadata
        result["summary"]["filename"] = filename
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing error: {str(e)}")


# Serve Frontend static assets
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_path = os.path.join(frontend_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
