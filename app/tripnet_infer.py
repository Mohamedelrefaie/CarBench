"""TripNet inference glue (registry mode ``"tripnet"``).

Two-stage per-car pipeline (models/tripnet — the from-scratch arXiv:2503.17400
reimplementation, see models/tripnet/IMPLEMENTATION_NOTES.md):

1. FIT: a learnable triplane (3x32x128x128) is optimised so the FROZEN shared
   occupancy decoder (results_from_scratch/tripnet_cache/triplane_cache/
   decoder.pt) reproduces the car's occupancy field — ray-parity labels via
   open3d RaycastingScene on the raw triangulated mesh, 70 Adam steps on 200k
   points drawn from a 1M-uniform + 1M-near-surface pool (~15-20 s on one
   GPU).  Fitted triplanes are cached per mesh-content hash.
2. QUERY: the frozen triplane goes through the shared U-Net + query MLP
   (results_from_scratch/tripnet/tripnet_from_scratch.pt) to predict
   normalized pressure at every vertex; denormalized with the global
   DrivAerNet moments.

Import note: models/tripnet uses flat module names (model/triplane/unet) that
collide with other models' files, so imports happen under a temporary
sys.path prefix and the flat names are purged from sys.modules afterwards
(class/function references are kept in the module-level state).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pyvista as pv
import torch

from app.model_registry import REPO_ROOT

_TRIPNET_DIR = os.path.join(REPO_ROOT, "models", "tripnet")
# Shared occupancy decoder: released alongside the TripNet weights.
_DECODER_CKPT = os.path.join(
    os.environ.get("CARBENCH_CHECKPOINT_DIR", os.path.join(REPO_ROOT, "checkpoints")),
    "tripnet_decoder.pt",
)
# Per-geometry triplane fits are cached here on first use (~20 s per design).
_SERVER_TRI_CACHE = os.environ.get(
    "CARBENCH_TRIPLANE_CACHE", os.path.join(REPO_ROOT, "cache", "triplanes")
)

# Global DrivAerNet pressure moments (models/tripnet/data_loader.py)
_P_MEAN = -93.41
_P_STD = 120.79

# Stage-2 fit recipe (models/tripnet/occupancy_fit.py cmd_stage2 defaults)
_FIT_STEPS = 70
_FIT_LR = 1e-3
_FIT_BATCH = 200_000
_N_UNIFORM = 1_000_000
_N_SURFACE = 1_000_000
_QUERY_CHUNK = 200_000

_STATE: Dict = {}


def _load(checkpoint_path: str, device: str) -> None:
    """Load the shared TripNet + occupancy decoder once (kept resident)."""
    if _STATE.get("device") == device and "model" in _STATE:
        return
    flat = ("model", "triplane", "unet", "data_loader", "utils")
    saved_path = sys.path.copy()
    saved_mods = {m: sys.modules.pop(m) for m in flat if m in sys.modules}
    sys.path.insert(0, _TRIPNET_DIR)
    try:
        import triplane as _tri
        import model as _tmodel
        from unet import OccupancyDecoder

        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = sd.get("config") or {}
        model = _tmodel.TripNet(
            channels=cfg.get("channels", 32), res=cfg.get("res", 128),
            unet_base=cfg.get("unet_base", 64), mlp_width=cfg.get("mlp_width", 256),
        )
        model.load_state_dict(sd["model"])
        model.eval().to(device)

        dec_ck = torch.load(_DECODER_CKPT, map_location="cpu", weights_only=False)
        decoder = OccupancyDecoder(in_dim=3 * cfg.get("channels", 32))
        decoder.load_state_dict(dec_ck["decoder"])
        decoder.eval().to(device)
        for p in decoder.parameters():
            p.requires_grad_(False)

        _STATE.update(
            model=model, decoder=decoder, device=device,
            sample_triplane=_tri.sample_triplane, TriplaneParam=_tri.TriplaneParam,
        )
    finally:
        sys.path = saved_path
        for m in flat:
            sys.modules.pop(m, None)
        sys.modules.update(saved_mods)


def clear() -> bool:
    """Drop the resident TripNet weights (mirrors inference_core cache hooks)."""
    had = "model" in _STATE
    _STATE.clear()
    if had and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return had


def _make_pool(surface: pv.PolyData, seed: int):
    """Occupancy pool straight from the in-memory mesh (occupancy_fit.make_pool,
    minus the .vtk round-trip): ray-parity labels tolerate non-watertight cars."""
    import open3d as o3d

    tri_mesh = surface.triangulate()
    v = np.asarray(tri_mesh.points, np.float32)
    fc = np.asarray(tri_mesh.faces).reshape(-1, 4)[:, 1:].astype(np.uint32)
    tm = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(fc))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)

    rng = np.random.default_rng(seed)
    lo, hi = v.min(0), v.max(0)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))
    qu = rng.uniform(lo - 0.03 * ext, hi + 0.03 * ext,
                     size=(_N_UNIFORM, 3)).astype(np.float32)
    idx = rng.integers(0, len(v), _N_SURFACE)
    qs = (v[idx] + rng.standard_normal((_N_SURFACE, 3)).astype(np.float32) * 0.01 * diag)
    pts = np.concatenate([qu, qs], 0)
    occ = scene.compute_occupancy(o3d.core.Tensor(pts), nthreads=8).numpy().astype(np.float32)
    center = (lo + hi) * 0.5
    scale = np.clip((hi - lo) * 0.5, 1e-6, None)
    pts_ndc = ((pts - center) / scale).astype(np.float32)
    return pts_ndc, occ, lo.astype(np.float32), hi.astype(np.float32)


def _fit_triplane(surface: pv.PolyData, device: str, seed: int):
    pool_ndc, occ, lo, hi = _make_pool(surface, seed)
    occ_frac = float(occ.mean())
    pool_pts = torch.from_numpy(pool_ndc).to(device)
    pool_occ = torch.from_numpy(occ).to(device)
    tri = _STATE["TriplaneParam"](32, 128).to(device)
    opt = torch.optim.Adam(tri.parameters(), lr=_FIT_LR)
    bce = torch.nn.BCEWithLogitsLoss()
    decoder = _STATE["decoder"]
    loss = None
    for _ in range(_FIT_STEPS):
        sel = torch.randint(0, pool_pts.shape[0], (_FIT_BATCH,), device=device)
        feat = tri(pool_pts[sel].unsqueeze(0)).squeeze(0)
        loss = bce(decoder(feat), pool_occ[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    del pool_pts, pool_occ
    return {
        "planes": tri.planes.detach().half().cpu(),
        "bbox_min": lo, "bbox_max": hi,
        "final_bce": float(loss) if loss is not None else None,
        "occ_frac": occ_frac,
    }


def predict_full(surface: pv.PolyData, checkpoint_path: str, device: str,
                 seed: int) -> Tuple[np.ndarray, Dict]:
    """Pressure at every vertex of ``surface`` (DrivAerNet meter frame)."""
    _load(checkpoint_path, device)
    pts = np.asarray(surface.points, np.float32)
    key = hashlib.sha1(np.ascontiguousarray(pts).tobytes()).hexdigest()[:24]
    cache_fp = os.path.join(_SERVER_TRI_CACHE, key + ".pt")

    fit_seconds = 0.0
    cached = os.path.exists(cache_fp)
    if cached:
        tp = torch.load(cache_fp, map_location="cpu", weights_only=False)
    else:
        t0 = time.time()
        tp = _fit_triplane(surface, device, seed)
        fit_seconds = time.time() - t0
        os.makedirs(_SERVER_TRI_CACHE, exist_ok=True)
        torch.save(tp, cache_fp)

    planes = tp["planes"].float().to(device)
    if planes.dim() == 4:
        planes = planes.unsqueeze(0)
    lo = np.asarray(tp["bbox_min"], np.float32)
    hi = np.asarray(tp["bbox_max"], np.float32)
    center = (lo + hi) * 0.5
    scale = np.clip((hi - lo) * 0.5, 1e-6, None)
    ndc = torch.from_numpy(((pts - center) / scale).astype(np.float32)).to(device)

    model = _STATE["model"]
    sample_triplane = _STATE["sample_triplane"]
    preds = []
    with torch.no_grad():
        refined = model.refine(planes)
        for i in range(0, ndc.shape[0], _QUERY_CHUNK):
            feat = sample_triplane(refined, ndc[i : i + _QUERY_CHUNK].unsqueeze(0))
            preds.append(model.mlp(feat).squeeze(0).float().cpu())
    full = (torch.cat(preds).numpy().astype(np.float32) * _P_STD) + _P_MEAN

    extra = {
        "triplane_cached": bool(cached),
        "fit_seconds": round(fit_seconds, 2),
        "final_bce": tp.get("final_bce"),
        "occ_fraction": (round(tp["occ_frac"], 3) if tp.get("occ_frac") is not None else None),
    }
    return full, extra
