import base64
import io
import os
import time
import uuid
import threading
import queue
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting, AutoPipelineForImage2Image

app = FastAPI(title="asset-pipeline image service")

MODEL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
MODEL_INPAINT = os.getenv("SDXL_INPAINT", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "960"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "640"))

_repo_root = Path(__file__).resolve().parents[1]
_out_dir = _repo_root / "outputs"
_out_dir.mkdir(parents=True, exist_ok=True)

_pipe_t2i = None
_pipe_i2i = None
_pipe_inpaint = None

# ControlNet caches
_controlnets: Dict[str, Any] = {}
_pipe_t2i_cn: Dict[tuple, Any] = {}
_pipe_i2i_cn: Dict[tuple, Any] = {}
_pipe_inpaint_cn: Dict[tuple, Any] = {}

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


def _load_i2i_once():
    global _pipe_i2i
    if _pipe_i2i is not None:
        return

    kwargs = {"torch_dtype": DTYPE}
    if DTYPE == torch.float16:
        try:
            _pipe_i2i = AutoPipelineForImage2Image.from_pretrained(MODEL_BASE, variant="fp16", **kwargs).to(DEVICE)
        except Exception:
            _pipe_i2i = AutoPipelineForImage2Image.from_pretrained(MODEL_BASE, **kwargs).to(DEVICE)
    else:
        _pipe_i2i = AutoPipelineForImage2Image.from_pretrained(MODEL_BASE, **kwargs).to(DEVICE)

    _try_enable_xformers(_pipe_i2i)


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


def _hex_to_rgb(hex_color: str):
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError("bg_color must be #RRGGBB")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _downscale_max(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_size:
        return img
    scale = max_size / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), resample=Image.LANCZOS)


# ----------------------------
# Conditioning plumbing
# ----------------------------
class ImageRef(BaseModel):
    url: Optional[str] = None
    b64: Optional[str] = None


class ControlInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    # Generic and composable. Provide a model_id per control.
    # type: openpose / canny / depth / lineart / silhouette
    type: str
    model_id: Optional[str] = None

    image_url: Optional[str] = None
    image_b64: Optional[str] = None

    weight: float = 1.0
    start: float = 0.0
    end: float = 1.0

    preprocess: bool = False
    processor_params: Optional[Dict[str, Any]] = None


class IdentityConditioning(BaseModel):
    # No-op placeholder layer (stores metadata for later InstantID/PhotoMaker).
    refs: List[ImageRef] = []
    strength: float = 1.0
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class ConditioningLayer(BaseModel):
    control_inputs: Optional[List[ControlInput]] = None
    identity: Optional[IdentityConditioning] = None



# ControlNet default model IDs (env-overridable)
CN_DEFAULT_CANNY      = os.getenv("CN_DEFAULT_CANNY",      "diffusers/controlnet-canny-sdxl-1.0")
CN_DEFAULT_DEPTH      = os.getenv("CN_DEFAULT_DEPTH",      "diffusers/controlnet-depth-sdxl-1.0")
CN_DEFAULT_OPENPOSE   = os.getenv("CN_DEFAULT_OPENPOSE",   "xinsir/controlnet-openpose-sdxl-1.0")
CN_DEFAULT_LINEART    = os.getenv("CN_DEFAULT_LINEART",    "")
CN_DEFAULT_SILHOUETTE = os.getenv("CN_DEFAULT_SILHOUETTE", CN_DEFAULT_CANNY)

def _default_controlnet_id(control_type: str):
    t = (control_type or "").lower()
    mapping = {
        "canny":      CN_DEFAULT_CANNY,
        "depth":      CN_DEFAULT_DEPTH,
        "openpose":   CN_DEFAULT_OPENPOSE,
        "lineart":    (CN_DEFAULT_LINEART or None),
        # silhouette maps are binary edges/masks; defaulting to canny CN is a practical baseline
        "silhouette": CN_DEFAULT_SILHOUETTE,
    }
    return mapping.get(t)


def _load_imgref(url: Optional[str], b64: Optional[str]) -> Image.Image:
    if (url is None) == (b64 is None):
        raise ValueError("provide exactly one of *_url or *_b64")
    img = _load_pil_from_url(url) if url else _load_pil_from_b64(b64)
    return img


def _preprocess_control(ci: ControlInput, src_rgba: Image.Image, width: int, height: int) -> Image.Image:
    """
    Returns an RGB control map resized to (width,height).
    If preprocess=False, treats input image as already-a-control-map.
    If preprocess=True, tries to generate a map based on ci.type.
    """
    params = ci.processor_params or {}

    if not ci.preprocess:
        return _prepare_image(src_rgba, width, height)

    t = (ci.type or "").lower()

    if t == "silhouette":
        rgba = src_rgba.convert("RGBA")
        if rgba.size != (width, height):
            rgba = rgba.resize((width, height), resample=Image.LANCZOS)
        alpha = rgba.split()[-1]
        thr = int(params.get("alpha_threshold", 10))
        mask = alpha.point(lambda a: 255 if a > thr else 0).convert("L")
        return mask.convert("RGB")

    if t == "canny":
        try:
            import numpy as np
            import cv2
        except Exception as e:
            raise RuntimeError(f"canny preprocess requires numpy + opencv-python-headless: {e}")
        rgb = _prepare_image(src_rgba, width, height)
        arr = np.array(rgb)
        low = int(params.get("low_threshold", 100))
        high = int(params.get("high_threshold", 200))
        edges = cv2.Canny(arr, low, high)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    if t in ("depth", "lineart", "openpose"):
        try:
            from controlnet_aux import MidasDetector, LineartDetector, OpenposeDetector
        except Exception as e:
            raise RuntimeError(f"{t} preprocess requires controlnet-aux: {e}")

        rgb = _prepare_image(src_rgba, width, height)

        if t == "depth":
            det = MidasDetector.from_pretrained("Intel/dpt-hybrid-midas")
            out = det(rgb).convert("RGB")
            if out.size != (width, height):
                out = out.resize((width, height), resample=Image.LANCZOS)
            return out
        if t == "lineart":
            det = LineartDetector.from_pretrained("lllyasviel/Annotators")
            out = det(rgb).convert("RGB")
            if out.size != (width, height):
                out = out.resize((width, height), resample=Image.LANCZOS)
            return out
        if t == "openpose":
            det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            out = det(rgb).convert("RGB")
            if out.size != (width, height):
                out = out.resize((width, height), resample=Image.LANCZOS)
            return out

    raise ValueError(f"unknown control type: {ci.type}")


def _load_controlnet_once(model_id: str):
    if model_id in _controlnets:
        return _controlnets[model_id]

    try:
        from diffusers import ControlNetModel
    except Exception as e:
        raise RuntimeError(f"diffusers ControlNetModel not available: {e}")

    def _try_load(use_safetensors: bool):

        kwargs = {"torch_dtype": DTYPE, "use_safetensors": use_safetensors}

        if DTYPE == torch.float16:

            try:

                return ControlNetModel.from_pretrained(model_id, variant="fp16", **kwargs).to(DEVICE)

            except Exception:

                return ControlNetModel.from_pretrained(model_id, **kwargs).to(DEVICE)

        return ControlNetModel.from_pretrained(model_id, **kwargs).to(DEVICE)


    try:

        cn = _try_load(True)

    except Exception:

        cn = _try_load(False)


    _controlnets[model_id] = cn
    return cn


def _get_cn_pipe(task: str, model_ids: List[str]):
    """
    task: txt2img / img2img / inpaint
    model_ids order matters and must match control_images/scales order.
    """
    key = tuple(model_ids)

    if task == "txt2img" and key in _pipe_t2i_cn:
        return _pipe_t2i_cn[key]
    if task == "img2img" and key in _pipe_i2i_cn:
        return _pipe_i2i_cn[key]
    if task == "inpaint" and key in _pipe_inpaint_cn:
        return _pipe_inpaint_cn[key]

    try:
        from diffusers.pipelines.controlnet import MultiControlNetModel
        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetInpaintPipeline,
        )
    except Exception as e:
        raise RuntimeError(f"SDXL ControlNet pipelines not available in your diffusers install: {e}")

    cns = [_load_controlnet_once(mid) for mid in model_ids]
    controlnet = cns[0] if len(cns) == 1 else MultiControlNetModel(cns)

    if task == "txt2img":
        _load_t2i_once()
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pipe(_pipe_t2i, controlnet=controlnet).to(DEVICE)
        except Exception:
            # Fallback: load directly from base
            kwargs = {"torch_dtype": DTYPE, "use_safetensors": True}
            if DTYPE == torch.float16:
                try:
                    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(MODEL_BASE, controlnet=controlnet, variant="fp16", **kwargs).to(DEVICE)
                except Exception:
                    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(MODEL_BASE, controlnet=controlnet, **kwargs).to(DEVICE)
            else:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(MODEL_BASE, controlnet=controlnet, **kwargs).to(DEVICE)

        _try_enable_xformers(pipe)
        _pipe_t2i_cn[key] = pipe
        return pipe

    if task == "img2img":
        _load_i2i_once()
        try:
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(_pipe_i2i, controlnet=controlnet).to(DEVICE)
        except Exception:
            kwargs = {"torch_dtype": DTYPE, "use_safetensors": True}
            if DTYPE == torch.float16:
                try:
                    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(MODEL_BASE, controlnet=controlnet, variant="fp16", **kwargs).to(DEVICE)
                except Exception:
                    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(MODEL_BASE, controlnet=controlnet, **kwargs).to(DEVICE)
            else:
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(MODEL_BASE, controlnet=controlnet, **kwargs).to(DEVICE)

        _try_enable_xformers(pipe)
        _pipe_i2i_cn[key] = pipe
        return pipe

    if task == "inpaint":
        _load_inpaint_once()
        try:
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pipe(_pipe_inpaint, controlnet=controlnet).to(DEVICE)
        except Exception:
            kwargs = {"torch_dtype": DTYPE, "use_safetensors": True}
            if DTYPE == torch.float16:
                try:
                    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(MODEL_INPAINT, controlnet=controlnet, variant="fp16", **kwargs).to(DEVICE)
                except Exception:
                    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(MODEL_INPAINT, controlnet=controlnet, **kwargs).to(DEVICE)
            else:
                pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(MODEL_INPAINT, controlnet=controlnet, **kwargs).to(DEVICE)

        _try_enable_xformers(pipe)
        _pipe_inpaint_cn[key] = pipe
        return pipe

    raise ValueError(f"unknown task: {task}")


def _identity_meta(identity: Optional[IdentityConditioning]) -> Optional[Dict[str, Any]]:
    if identity is None:
        return None
    return {
        "strength": identity.strength,
        "method": identity.method,
        "params": identity.params or {},
        "num_refs": len(identity.refs or []),
    }


# ----------------------------
# Requests
# ----------------------------
class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    steps: int = 30
    guidance: float = 5.0
    seed: Optional[int] = None

    conditioning: Optional[ConditioningLayer] = None


class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None

    init_image_url: Optional[str] = None
    init_image_b64: Optional[str] = None

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    steps: int = 30
    guidance: float = 5.0
    strength: float = 0.6
    seed: Optional[int] = None

    conditioning: Optional[ConditioningLayer] = None


class InpaintRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None

    init_image_url: Optional[str] = None
    init_image_b64: Optional[str] = None

    mask_image_url: Optional[str] = None
    mask_image_b64: Optional[str] = None

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    steps: int = 30
    guidance: float = 5.0
    strength: float = 0.75
    seed: Optional[int] = None

    conditioning: Optional[ConditioningLayer] = None


class CleanupRequest(BaseModel):
    image_url: Optional[str] = None
    image_b64: Optional[str] = None

    # "solid_bg" = composite on a solid background for Trellis readiness
    # "remove_bg" = alpha cutout (requires rembg installed later)
    mode: str = "solid_bg"

    bg_color: str = "#FFFFFF"
    max_size: int = 2048


def _run_txt2img(job_id: str, req: Txt2ImgRequest):
    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

    meta: Dict[str, Any] = {}
    controls = (req.conditioning.control_inputs if req.conditioning and req.conditioning.control_inputs else []) or []
    identity = req.conditioning.identity if req.conditioning else None
    imeta = _identity_meta(identity)
    if imeta:
        meta["identity"] = imeta

    if not controls:
        _load_t2i_once()
        out = _pipe_t2i(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=gen,
        )
    else:
        # ControlNet path
        model_ids: List[str] = []
        control_images: List[Image.Image] = []
        scales: List[float] = []
        starts: List[float] = []
        ends: List[float] = []

        for ci in controls:
            mid = ci.model_id or _default_controlnet_id(ci.type)
            if not mid:
                raise ValueError("each control_inputs item requires model_id")
            ci.model_id = mid
            model_ids.append(mid)
            if (ci.image_url is None) == (ci.image_b64 is None):
                raise ValueError("txt2img control_inputs requires exactly one of image_url or image_b64 per control")
            src = _load_imgref(ci.image_url, ci.image_b64)
            cmap = _preprocess_control(ci, src, req.width, req.height)
            control_images.append(cmap)
            scales.append(float(ci.weight))
            starts.append(float(ci.start))
            ends.append(float(ci.end))

        pipe = _get_cn_pipe("txt2img", model_ids)

        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=gen,
            image=control_images if len(control_images) > 1 else control_images[0],
            controlnet_conditioning_scale=scales if len(scales) > 1 else scales[0],
            control_guidance_start=starts if len(starts) > 1 else starts[0],
            control_guidance_end=ends if len(ends) > 1 else ends[0],
        )

        meta["controls"] = [{"type": c.type, "model_id": c.model_id, "weight": c.weight, "start": c.start, "end": c.end, "preprocess": c.preprocess} for c in controls]

    png_path = _out_dir / f"{job_id}.png"
    out.images[0].save(png_path, format="PNG")
    return png_path, meta


def _run_img2img(job_id: str, req: Img2ImgRequest):
    if (req.init_image_url is None) == (req.init_image_b64 is None):
        raise ValueError("provide exactly one of init_image_url or init_image_b64")

    init_rgba = _load_imgref(req.init_image_url, req.init_image_b64).convert("RGBA")
    init_img = _prepare_image(init_rgba, req.width, req.height)

    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

    meta: Dict[str, Any] = {}
    controls = (req.conditioning.control_inputs if req.conditioning and req.conditioning.control_inputs else []) or []
    identity = req.conditioning.identity if req.conditioning else None
    imeta = _identity_meta(identity)
    if imeta:
        meta["identity"] = imeta

    if not controls:
        _load_i2i_once()
        out = _pipe_i2i(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=init_img,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            strength=req.strength,
            generator=gen,
        )
    else:
        model_ids: List[str] = []
        control_images: List[Image.Image] = []
        scales: List[float] = []
        starts: List[float] = []
        ends: List[float] = []

        for ci in controls:
            mid = ci.model_id or _default_controlnet_id(ci.type)
            if not mid:
                raise ValueError("each control_inputs item requires model_id")
            ci.model_id = mid
            model_ids.append(mid)

            # If control image omitted and preprocess=True, use init image as source.
            if ci.image_url is None and ci.image_b64 is None:
                if not ci.preprocess:
                    raise ValueError("img2img control_inputs requires image_url/image_b64 unless preprocess=true")
                src = init_rgba
            else:
                src = _load_imgref(ci.image_url, ci.image_b64)

            cmap = _preprocess_control(ci, src, req.width, req.height)
            control_images.append(cmap)
            scales.append(float(ci.weight))
            starts.append(float(ci.start))
            ends.append(float(ci.end))

        pipe = _get_cn_pipe("img2img", model_ids)

        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=init_img,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            strength=req.strength,
            generator=gen,
            control_image=control_images if len(control_images) > 1 else control_images[0],
            controlnet_conditioning_scale=scales if len(scales) > 1 else scales[0],
            control_guidance_start=starts if len(starts) > 1 else starts[0],
            control_guidance_end=ends if len(ends) > 1 else ends[0],
        )

        meta["controls"] = [{"type": c.type, "model_id": c.model_id, "weight": c.weight, "start": c.start, "end": c.end, "preprocess": c.preprocess} for c in controls]

    png_path = _out_dir / f"{job_id}.png"
    out.images[0].save(png_path, format="PNG")
    return png_path, meta


def _run_inpaint(job_id: str, req: InpaintRequest):
    if (req.init_image_url is None) == (req.init_image_b64 is None):
        raise ValueError("provide exactly one of init_image_url or init_image_b64")
    if (req.mask_image_url is None) == (req.mask_image_b64 is None):
        raise ValueError("provide exactly one of mask_image_url or mask_image_b64")

    init_rgba = _load_imgref(req.init_image_url, req.init_image_b64).convert("RGBA")
    init_img = _prepare_image(init_rgba, req.width, req.height)

    mask_img = _load_imgref(req.mask_image_url, req.mask_image_b64)
    mask_img = _prepare_mask(mask_img, req.width, req.height)

    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(req.seed)

    meta: Dict[str, Any] = {}
    controls = (req.conditioning.control_inputs if req.conditioning and req.conditioning.control_inputs else []) or []
    identity = req.conditioning.identity if req.conditioning else None
    imeta = _identity_meta(identity)
    if imeta:
        meta["identity"] = imeta

    if not controls:
        _load_inpaint_once()
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
    else:
        model_ids: List[str] = []
        control_images: List[Image.Image] = []
        scales: List[float] = []
        starts: List[float] = []
        ends: List[float] = []

        for ci in controls:
            mid = ci.model_id or _default_controlnet_id(ci.type)
            if not mid:
                raise ValueError("each control_inputs item requires model_id")
            ci.model_id = mid
            model_ids.append(mid)

            if ci.image_url is None and ci.image_b64 is None:
                if not ci.preprocess:
                    raise ValueError("inpaint control_inputs requires image_url/image_b64 unless preprocess=true")
                src = init_rgba
            else:
                src = _load_imgref(ci.image_url, ci.image_b64)

            cmap = _preprocess_control(ci, src, req.width, req.height)
            control_images.append(cmap)
            scales.append(float(ci.weight))
            starts.append(float(ci.start))
            ends.append(float(ci.end))

        pipe = _get_cn_pipe("inpaint", model_ids)

        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=init_img,
            mask_image=mask_img,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            strength=req.strength,
            generator=gen,
            control_image=control_images if len(control_images) > 1 else control_images[0],
            controlnet_conditioning_scale=scales if len(scales) > 1 else scales[0],
            control_guidance_start=starts if len(starts) > 1 else starts[0],
            control_guidance_end=ends if len(ends) > 1 else ends[0],
        )

        meta["controls"] = [{"type": c.type, "model_id": c.model_id, "weight": c.weight, "start": c.start, "end": c.end, "preprocess": c.preprocess} for c in controls]

    png_path = _out_dir / f"{job_id}.png"
    out.images[0].save(png_path, format="PNG")
    return png_path, meta


def _run_cleanup(job_id: str, req: CleanupRequest):
    if (req.image_url is None) == (req.image_b64 is None):
        raise ValueError("provide exactly one of image_url or image_b64")

    img = _load_imgref(req.image_url, req.image_b64).convert("RGBA")
    img = _downscale_max(img, req.max_size)

    meta: Dict[str, Any] = {"mode": req.mode}

    if req.mode == "solid_bg":
        r, g, b = _hex_to_rgb(req.bg_color)
        bg = Image.new("RGBA", img.size, (r, g, b, 255))
        out = Image.alpha_composite(bg, img).convert("RGB")
        meta["bg_color"] = req.bg_color
    elif req.mode == "remove_bg":
        try:
            from rembg import remove
        except Exception as e:
            raise RuntimeError(f"remove_bg requires rembg to be installed: {e}")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_bytes = remove(buf.getvalue())
        out = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    else:
        raise ValueError("mode must be solid_bg or remove_bg")

    png_path = _out_dir / f"{job_id}.png"
    out.save(png_path, format="PNG")
    return png_path, meta


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

        # GPU serialize (keeps VRAM stable). cleanup is cheap but kept serialized for simplicity.
        with _gpu_lock:
            if kind == "txt2img":
                png_path, meta = _run_txt2img(job_id, req)
            elif kind == "img2img":
                png_path, meta = _run_img2img(job_id, req)
            elif kind == "inpaint":
                png_path, meta = _run_inpaint(job_id, req)
            elif kind == "cleanup":
                png_path, meta = _run_cleanup(job_id, req)
            else:
                raise ValueError(f"unknown job kind: {kind}")

        with _lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["finished_at"] = time.time()
            _jobs[job_id]["png_path"] = str(png_path)
            _jobs[job_id]["meta"] = meta

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
    return {
        "ok": True,
        "device": DEVICE,
        "model": MODEL_BASE,
        "inpaint_model": MODEL_INPAINT,
        "dtype": str(DTYPE),
    }


@app.post("/txt2img_job")
def txt2img_job(req: Txt2ImgRequest):
    _ensure_worker()
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"kind": "txt2img", "status": "queued", "created_at": time.time(), "request": req}
    _q.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/img2img_job")
def img2img_job(req: Img2ImgRequest):
    _ensure_worker()
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"kind": "img2img", "status": "queued", "created_at": time.time(), "request": req}
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


@app.post("/cleanup_job")
def cleanup_job(req: CleanupRequest):
    _ensure_worker()
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"kind": "cleanup", "status": "queued", "created_at": time.time(), "request": req}
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