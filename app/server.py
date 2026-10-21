"""FastAPI inference server (cluster side).

Bound to 127.0.0.1 and reached from the Windows client over an SSH tunnel. Each
/predict request loads the chosen model on demand, normalizes the uploaded STL to
the DrivAerNet frame, predicts the surface-pressure field, and streams back a VTP
(the car surface with a `pressure` point-scalar). Prediction metadata + the
normalization report are returned in the ``X-Prediction-Meta`` response header.

Run (on the cluster, pick one GPU):
    cd /path/to/carbench && source venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

Then from Windows: ssh -L 8000:127.0.0.1:8000 user@cluster, and point the app at
http://localhost:8000 .
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback

import pyvista as pv
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app import inference_core, model_registry
from app.inference_core import predict_on_mesh
from app.stl_normalize import normalize_to_drivaernet

app = FastAPI(title="DrivAerNet Pressure Inference", version="0.2.0")

# v1: serialize GPU work — one inference at a time.
_GPU_LOCK = threading.Lock()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def _prewarm() -> None:
    """Pre-load the default model(s) so the first user prediction pays neither
    the model load nor cold-CUDA init.  Set ``PREWARM_MODELS`` (comma-separated)
    to change the warm set; empty string disables pre-warming."""
    spec = os.environ.get("PREWARM_MODELS", "transolver")
    for name in (n.strip() for n in spec.split(",") if n.strip()):
        try:
            with _GPU_LOCK:
                inference_core.prewarm(name, DEVICE)
            print(f"[prewarm] loaded {name}", flush=True)
        except Exception as e:  # never block startup on a warm failure
            print(f"[prewarm] {name} failed: {e}", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cached_models": inference_core.cached_models(),
    }


@app.post("/cache/clear")
def cache_clear():
    """Drop all resident models and free their VRAM."""
    inference_core.clear_cache()
    return {"status": "ok", "cached_models": inference_core.cached_models()}


@app.get("/models")
def models():
    return {"models": model_registry.list_models(only_available=True)}


@app.post("/predict")
def predict(
    stl: UploadFile = File(...),
    model: str = Form(...),
    axis_override: str = Form("{}"),
    scale_to_canonical: bool = Form(True),
    frame_mode: str = Form("auto"),
    strategy: str = Form("decimate"),
):
    """STL + model name -> VTP pressure field (body) + JSON meta (header).

    ``strategy``: ``decimate`` (fast: sample ~10k + interpolate) or ``patch``
    (accurate: cover every point with K passes, no interpolation).
    """
    try:
        model_registry.get(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    strategy = (strategy or "decimate").strip().lower()
    if strategy not in ("decimate", "patch"):
        raise HTTPException(status_code=400, detail="strategy must be 'decimate' or 'patch'")

    try:
        overrides = json.loads(axis_override or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="axis_override must be valid JSON")

    raw = stl.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty STL upload")

    tmp_stl = tmp_vtp = None
    t0 = time.time()
    try:
        suffix = os.path.splitext(stl.filename or "car.stl")[1] or ".stl"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw)
            tmp_stl = f.name

        mesh = pv.read(tmp_stl)
        aligned, report = normalize_to_drivaernet(
            mesh, axis_override=overrides, scale_to_canonical=scale_to_canonical,
            frame_mode=frame_mode,
        )
        t_norm = time.time()

        with _GPU_LOCK:
            out, meta = predict_on_mesh(model, aligned, device=DEVICE, strategy=strategy)
        t_pred = time.time()

        with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
            tmp_vtp = f.name
        out.save(tmp_vtp)
        with open(tmp_vtp, "rb") as f:
            vtp_bytes = f.read()

        meta["timings_s"] = {
            "normalize": round(t_norm - t0, 3),
            "inference": round(t_pred - t_norm, 3),
            "total": round(time.time() - t0, 3),
        }
        meta["normalize_report"] = report
        headers = {
            "X-Prediction-Meta": json.dumps(meta),
            "Content-Disposition": 'attachment; filename="prediction.vtp"',
        }
        return Response(content=vtp_bytes, media_type="application/octet-stream", headers=headers)

    except HTTPException:
        raise
    except Exception as e:  # surface a readable error to the client
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (tmp_stl, tmp_vtp):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
