"""
main.py — FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analyzer.perplexity import load_model
from scorer import analyze_essay


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load GPT-2 on server launch
    print("[startup] Pre-loading GPT-2 model...")
    load_model()
    print("[startup] Application ready.")
    yield


app = FastAPI(
    title="AI Admissions Essay Detector API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=15000, description="Essay text to analyze")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Essay text cannot be empty.")
    
    result = analyze_essay(req.text)
    return result
