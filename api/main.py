"""EssayTrace API — feature measurement in, calibrated scores out.

There is no path from this module (or anything it imports on the
request path) to a chat/instruction model. Detection is features + a
fitted classifier. See scripts/generate_dataset.py for the separate
data-generation path.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.detect import detect, load_bundle
from api.schemas import DetectRequest, DetectResponse

app = FastAPI(
    title="EssayTrace",
    description="Sentence-level AI-writing signals for admissions essays. Not a verdict.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    bundle = load_bundle()
    return {
        "ok": True,
        "trained_classifier": bundle is not None,
        "model_name": (bundle or {}).get("model_name", "heuristic-fallback"),
    }


@app.post("/api/detect", response_model=DetectResponse)
def api_detect(req: DetectRequest) -> DetectResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Paste an essay first.")
    if len(text) < 40:
        raise HTTPException(
            status_code=400,
            detail="Need a longer passage (at least a few sentences) to measure honestly.",
        )
    return detect(text)
