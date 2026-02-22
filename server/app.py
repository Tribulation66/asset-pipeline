import base64
import io
import os
import time
import uuid
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image

app = FastAPI(title="asset-pipeline image service")

MODEL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_repo_root = Path(__file__).resolve().parents[1]
_out_dir = _repo_root / "outputs"
_out_dir.mkdir(parents=True, exist_ok=True)

_pipe = None

_jobs: Dict[str, Dict[str, Any]] = {}
_q: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_lock = threading.Lock()


def _try_enable_xformers(pipe):
    if DEVICE != "cuda":
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def _load_pipe_once():
    global _pipe
    if _pipe is not None:
        return

    kwargs = {"torch_dtype": DTYPE}
    if DTYPE == torch.float16:
        try:
            _pipe = AutoPipelineForText2Image.from_pretrained(MODEL_BASE, variant="fp16", **kwargs).to(DEVICE)
        except Exception:
            _pipe = AutoPipelineForText2Image.from_pretrained(MODEL_BASE, **kwargs).to(DEVICE)
    else:
        _pipe = AutoPipelineForText2Image.from_pretrained(MODEL_BASE, **kwargs).to(DEVICE)

    _try_enable_xformers(_pipe)


class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 5.0
    seed: Optional[int] = None


def _run_job(job_id: str, req: Txt2ImgRequest):
    try:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()

        _load_pipe_once()

        gen = None
        if req.seed is not None:
            gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

        out = _pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=gen,
        )

        png_path = _out_dir / f"{job_id}.png"
        out.images[0].save(png_path, format="PNG")

        with _lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["finished_at"] = time.time()
            _jobs[job_id]["png_path"] = str(png_path)

    except Exception as e:
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["finished_at"] = time.time()
            _jobs[job_id]["error"] = repr(e)


def _worker_loop():
    while True:
        job_id = _q.get()
        with _lock:
            req = _jobs.get(job_id, {}).get("request")
        if req is None:
            continue
        _run_job(job_id, req)


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()
    _worker_started = True


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_BASE}


@app.post("/txt2img_job")
def txt2img_job(req: Txt2ImgRequest):
    _ensure_worker()
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "request": req,
        }
    _q.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job not found")
        # Don't echo full request back; keep response small
        resp = {k: v for k, v in j.items() if k != "request"}
    # Provide a stable relative path for the image if done
    if resp.get("status") == "done":
        resp["image_endpoint"] = f"/jobs/{job_id}/image.png"
    return resp


@app.get("/jobs/{job_id}/image.png")
def job_image(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job not found")
        if j.get("status") != "done":
            raise HTTPException(status_code=409, detail=f"job not done (status={j.get('status')})")
        png_path = j.get("png_path")
    if not png_path or not Path(png_path).exists():
        raise HTTPException(status_code=500, detail="image missing")
    return FileResponse(png_path, media_type="image/png")