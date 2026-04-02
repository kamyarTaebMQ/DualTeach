"""
Generic FastAPI Inference Service (DualTeach / Teacher)
=========================================================
Wraps any HuggingFace-compatible model as a REST API for benchmarking.

Endpoints:
  POST /solve   — submit a math problem, receive the model's answer
  GET  /health  — liveness probe
  GET  /info    — service metadata (includes actual GPU memory)

Usage:
    # Local checkpoint:
    python serve.py --checkpoint checkpoints/student/epoch_5 --port 8000

    # HuggingFace model ID:
    python serve.py --model_id google/gemma-3-12b-it --model_name "Teacher-12B" --port 8000
"""

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.environ.get("SERVE_CHECKPOINT", "checkpoints/student/epoch_5")
DEFAULT_MODEL_ID   = os.environ.get("SERVE_MODEL_ID", "")
DEFAULT_MODEL_NAME = os.environ.get("SERVE_MODEL_NAME", "model")

# ══════════════════════════════════════════════════════════════════════════════
#  Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class MathRequest(BaseModel):
    problem: str = Field(
        ...,
        description="The math word problem to solve.",
        examples=["Janet's ducks lay 16 eggs per day. She eats 3 for breakfast. How many eggs does she have left?"],
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate (1–512).",
    )


class MathResponse(BaseModel):
    answer:             str
    inference_time_ms:  float
    model_name:         str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


class InfoResponse(BaseModel):
    model:      str
    vram_gb:    float
    source:     str


# ══════════════════════════════════════════════════════════════════════════════
#  Global model state
# ══════════════════════════════════════════════════════════════════════════════

_model      = None
_tokenizer  = None
_model_src  = DEFAULT_CHECKPOINT
_model_name = DEFAULT_MODEL_NAME
_vram_gb    = 0.0


def _load_model(source: str, model_name: str):
    """Load model + tokenizer from a local checkpoint or HuggingFace hub ID."""
    global _model, _tokenizer, _vram_gb, _model_src, _model_name
    _model_src  = source
    _model_name = model_name
    print(f"[serve] Loading '{model_name}' from '{source}' ...")
    _tokenizer = AutoTokenizer.from_pretrained(source)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    max_vram = int(os.environ.get("SERVE_MAX_VRAM", "0") or "0")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if max_vram > 0:
        kwargs["max_memory"] = {0: f"{max_vram}GiB", "cpu": "0GiB"}
        print(f"[serve] VRAM cap: {max_vram} GiB (cpu offload disabled)")

    try:
        _model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except Exception as e:
        print(f"[serve] LOAD FAILED: {e}", flush=True)
        os._exit(1)

    _model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

    print(f"[serve] Model loaded. VRAM used: {_vram_gb} GB. Service is ready.")


# ══════════════════════════════════════════════════════════════════════════════
#  App lifespan (startup / shutdown)
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    src  = os.environ.get("SERVE_CHECKPOINT", DEFAULT_CHECKPOINT)
    name = os.environ.get("SERVE_MODEL_NAME", DEFAULT_MODEL_NAME)
    _load_model(src, name)
    yield
    global _model, _tokenizer
    del _model, _tokenizer
    _model = _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[serve] Model unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="LLM Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ── /solve ────────────────────────────────────────────────────────────────────

@app.post("/solve", response_model=MathResponse, summary="Solve a math word problem")
async def solve_math(request: MathRequest):
    """
    Generate an answer to a math word problem.

    The model uses greedy decoding for deterministic, reproducible output.
    Returns the generated answer along with server-side inference latency.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded — service unavailable.")

    prompt = f"Problem: {request.problem}\nAnswer:"
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    answer = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return MathResponse(
        answer=answer.strip(),
        inference_time_ms=round(elapsed_ms, 2),
        model_name=_model_name,
    )


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    """Returns service health status and whether the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )


# ── /info ─────────────────────────────────────────────────────────────────────

@app.get("/info", response_model=InfoResponse, summary="Service metadata")
async def info():
    """Returns model metadata including actual measured VRAM usage."""
    return InfoResponse(
        model=_model_name,
        vram_gb=_vram_gb,
        source=_model_src,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generic FastAPI LLM Inference Service")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Local checkpoint directory path.")
    p.add_argument("--model_id", type=str, default="",
                   help="HuggingFace hub model ID (overrides --checkpoint).")
    p.add_argument("--model_name", type=str, default="model",
                   help="Display name for this model in responses.")
    p.add_argument("--host",     type=str, default="0.0.0.0")
    p.add_argument("--port",     type=int, default=8000)
    p.add_argument("--max_vram", type=int, default=0,
                   help="Cap GPU memory in GiB (0 = no cap). CPU offload disabled.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    source = args.model_id if args.model_id else args.checkpoint
    if not source:
        source = DEFAULT_MODEL_ID if DEFAULT_MODEL_ID else DEFAULT_CHECKPOINT

    # Pass config via env vars so the lifespan reads them after uvicorn forks
    os.environ["SERVE_CHECKPOINT"] = source
    os.environ["SERVE_MODEL_NAME"] = args.model_name or DEFAULT_MODEL_NAME
    if args.max_vram > 0:
        os.environ["SERVE_MAX_VRAM"] = str(args.max_vram)

    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="warning",
    )
