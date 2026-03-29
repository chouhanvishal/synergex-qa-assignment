"""
Synergex Med — AI-Powered Call QA System
POST /analyze-call  — Analyze a single call transcript
POST /batch-analyze — Analyze multiple transcripts
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.analyzer import analyze_transcript
from app.logger import align_uvicorn_log_format, get_logger
from app.schema import BatchRequest, BatchResponse, CallTranscript, QAAnalysis

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    align_uvicorn_log_format()
    logger.info("Synergex Med QA API starting up")
    yield
    logger.info("Synergex Med QA API shutting down")


app = FastAPI(
    title="Synergex Med — Call QA API",
    description="AI-powered quality analysis for clinical call center transcripts",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "synergex-qa"}


@app.post("/analyze-call", response_model=QAAnalysis)
async def analyze_call(payload: CallTranscript):
    """
    Receive a call transcript and return a structured quality analysis.
    """
    start = time.monotonic()
    logger.info(
        "analyze_call called",
        extra={"call_id": payload.call_id, "agent": payload.agent_name, "department": payload.department},
    )

    try:
        result = await analyze_transcript(payload)
    except Exception as exc:
        logger.error("Analysis failed for call_id=%s: %s", payload.call_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    elapsed = time.monotonic() - start
    logger.info("analyze_call completed in %.2fs for call_id=%s", elapsed, payload.call_id)
    return result


@app.post("/batch-analyze", response_model=BatchResponse)
async def batch_analyze(payload: BatchRequest):
    """
    Analyze a list of transcripts concurrently and return results for all.
    """
    tasks = [analyze_transcript(call) for call in payload.calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    analyses = []
    errors = []
    for call, result in zip(payload.calls, results):
        if isinstance(result, Exception):
            logger.error("Batch analysis failed for call_id=%s: %s", call.call_id, result)
            errors.append({"call_id": call.call_id, "error": str(result)})
        else:
            analyses.append(result)

    return BatchResponse(results=analyses, errors=errors)
