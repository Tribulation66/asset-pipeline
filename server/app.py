import base64
import io
import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image

app = FastAPI(title="asset-pipeline image service")

MODEL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_pipe = None


def _try_enable_xformers(pipe):
    if DEVICE != "cuda":
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        # xformers may be unavailable on some images; continue without it
        pass


def _load_pipe_once():
    global _pipe
    if _pipe is not None:
        return

    kwargs = {"torch_dtype": DTYPE}
    # fp16 variant exists for some model repos; fall back if not present
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


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_BASE}


@app.post("/txt2img")
def txt2img(req: Txt2ImgRequest):
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

    buf = io.BytesIO()
    out.images[0].save(buf, format="PNG")
    return {
        "seed": req.seed,
        "png_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }