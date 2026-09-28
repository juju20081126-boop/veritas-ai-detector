"""
FastAPI Server for Veritas AI Originality & AI Writing Detector
Provides REST APIs for real-time document analysis, multi-format file uploads,
PDF inspection report generation, and sample demonstrations.
"""

import os
import io
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.engine import AIDetectorEngine
from backend.document_parser import parse_uploaded_file
from backend.pdf_report import generate_ai_pdf_report
from samples.sample_data import SAMPLES

# Initialize FastAPI application
app = FastAPI(
    title="Veritas AI Detection Server",
    description="Turnitin-grade AI writing detection and originality audit engine",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine reference (initialized on startup)
engine: Optional[AIDetectorEngine] = None


@app.on_event("startup")
def startup_event():
    global engine
    engine = AIDetectorEngine.get_instance()


class DetectRequest(BaseModel):
    text: str
    filename: Optional[str] = "Pasted Text"


@app.get("/api/health")
def health_check():
    global engine
    is_ready = engine is not None
    return {
        "status": "online" if is_ready else "initializing",
        "engine": "Veritas AI Ensemble (RoBERTa + GPT2 + Stylometrics)",
        "device": engine.device if engine else "unknown",
        "timestamp": time.time()
    }


@app.get("/api/samples")
def get_samples():
    return {
        key: {
            "title": data["title"],
            "description": data["description"],
            "text": data["text"]
        }
        for key, data in SAMPLES.items()
    }


@app.post("/api/detect")
def detect_text(req: DetectRequest):
    global engine
    if not engine:
        engine = AIDetectorEngine.get_instance()

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text.split()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text is too brief for statistical confidence. Please provide at least 15-20 words."
        )

    try:
        result = engine.analyze_document(text, filename=req.filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global engine
    if not engine:
        engine = AIDetectorEngine.get_instance()

    filename = file.filename or "uploaded_document"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        extracted_text, metadata = parse_uploaded_file(filename, contents)
        extracted_text = extracted_text.strip()
        if not extracted_text or len(extracted_text.split()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Extracted document text is too brief for statistical confidence."
            )

        result = engine.analyze_document(extracted_text, filename=filename)
        result["metadata"] = metadata
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


class ReportRequest(BaseModel):
    analysis: dict


@app.post("/api/report")
def export_pdf_report(req: ReportRequest):
    try:
        pdf_bytes = generate_ai_pdf_report(req.analysis)
        filename = f"veritas_report_{int(time.time())}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_bytes))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {str(e)}")


@app.post("/api/comparative-audit")
def comparative_audit_endpoint(req: DetectRequest):
    global engine
    if not engine:
        engine = AIDetectorEngine.get_instance()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        from backend.reverse_engineered_detectors import run_full_comparative_audit
        results = run_full_comparative_audit(text, engine)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparative audit failed: {str(e)}")



# Serve Frontend static assets
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_path = os.path.join(frontend_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
