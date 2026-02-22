import base64
import io
import os
import time
import uuid
import threading
import queue
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting

app = FastAPI(title="asset-pipeline image service")

MODEL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
MODEL_INPAINT = os.getenv("SDXL_INPAINT", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_repo_root = Path(__file__).resolve().parents[1]
_out_dir = _repo_root / "outputs"
_out_dir.mkdir(parents=True, exist_ok=True)

_pipe_t2i = None
_pipe_inpaint = None

_jobs: Dict[str, Dict[str, Any]] = {}
_q: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_lock = threading.Lock()
_gpu_lock = threading.Lock()


def _try_enable_xformers(pipe):
    if DEVICE != "cuda":
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def _load_t2i_once():
    global _pipe_t2i
    if _pipe_t2i is not None:
        return

    kwargs = {"torch_dtype": DTYPE}
    if DTYPE == torch.float16:
        try:
            _pipe_t2i = AutoPipelineForText2Image.from_pretrained(MODEL_BASE, variant="fp16", **kwargs).to(DEVICE)
        except Exception:
            _pipe_t2i = AutoPipelineForText2Image.from_pretrained(MODEL_BASE, **kwargs).to(DEVICE)
    else:
        _pipe_t2i = AutoPipelineForText2Image.from_pretrained(MODEL_BASE, **kwargs).to(DEVICE)

    _try_enable_xformers(_pipe_t2i)


def _load_inpaint_once():
    global _pipe_inpaint
    if _pipe_inpaint is not None:
        return

    kwargs = {"torch_dtype": DTYPE}
    if DTYPE == torch.float16:
        try:
            _pipe_inpaint = AutoPipelineForInpainting.from_pretrained(MODEL_INPAINT, variant="fp16", **kwargs).to(DEVICE)
        except Exception:
            _pipe_inpaint = AutoPipelineForInpainting.from_pretrained(MODEL_INPAINT, **kwargs).to(DEVICE)
    else:
        _pipe_inpaint = AutoPipelineForInpainting.from_pretrained(MODEL_INPAINT, **kwargs).to(DEVICE)

    _try_enable_xformers(_pipe_inpaint)


def _load_pil_from_url(url: str) -> Image.Image:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("only http/https URLs are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "asset-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def _load_pil_from_b64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def _prepare_image(img: Image.Image, width: int, height: int) -> Image.Image:
    img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), resample=Image.LANCZOS)
    return img


def _prepare_mask(mask: Image.Image, width: int, height: int) -> Image.Image:
    # white = inpaint, black = keep
    mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=Image.NEAREST)
    return mask


class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 5.0
    seed: Optional[int] = None


class InpaintRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None

    init_image_url: Optional[str] = None
    init_image_b64: Optional[str] = None

    mask_image_url: Optional[str] = None
    mask_image_b64: Optional[str] = None

    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 5.0
    strength: float = 0.75
    seed: Optional[int] = None


def _run_txt2img(job_id: str, req: Txt2ImgRequest):
    _load_t2i_once()

    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

    out = _pipe_t2i(
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
    return png_path


def _run_inpaint(job_id: str, req: InpaintRequest):
    _load_inpaint_once()

    if (req.init_image_url is None) == (req.init_image_b64 is None):
        raise ValueError("provide exactly one of init_image_url or init_image_b64")
    if (req.mask_image_url is None) == (req.mask_image_b64 is None):
        raise ValueError("provide exactly one of mask_image_url or mask_image_b64")

    init_img = _load_pil_from_url(req.init_image_url) if req.init_image_url else _load_pil_from_b64(req.init_image_b64)
    mask_img = _load_pil_from_url(req.mask_image_url) if req.mask_image_url else _load_pil_from_b64(req.mask_image_b64)

    init_img = _prepare_image(init_img, req.width, req.height)
    mask_img = _prepare_mask(mask_img, req.width, req.height)

    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

    out = _pipe_inpaint(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        image=init_img,
        mask_image=mask_img,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        strength=req.strength,
        generator=gen,
    )

    png_path = _out_dir / f"{job_id}.png"
    out.images[0].save(png_path, format="PNG")
    return png_path


def _run_job(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            return
        kind = j.get("kind")
        req = j.get("request")

    try:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()

        with _gpu_lock:
            if kind == "txt2img":
                png_path = _run_txt2img(job_id, req)
            elif kind == "inpaint":
                png_path = _run_inpaint(job_id, req)
            else:
                raise ValueError(f"unknown job kind: {kind}")

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
        _run_job(job_id)


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()
    _worker_started = True


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_BASE, "inpaint_model": MODEL_INPAINT}


@app.post("/txt2img_job")
def txt2img_job(req: Txt2ImgRequest):
    _ensure_worker()
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"kind": "txt2img", "status": "queued", "created_at": time.time(), "request": req}
    _q.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/inpaint_job")
def inpaint_job(req: InpaintRequest):
    _ensure_worker()
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"kind": "inpaint", "status": "queued", "created_at": time.time(), "request": req}
    _q.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job not found")
        resp = {k: v for k, v in j.items() if k != "request"}
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