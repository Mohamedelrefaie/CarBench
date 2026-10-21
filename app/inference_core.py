"""Ground-truth-free inference core for the STL pressure app.

Reuses the validated single-car inference path (``single_car_inference_fixed.py``):
the same wrapper loading, 6D/point-cloud feature construction, normalization, and
full-mesh interpolation that produced the paper figures. The only additions here
are (a) a sampler that does not require a ground-truth ``p`` field (STL is geometry
only) and (b) an on-demand load→predict→free wrapper.

Special dispatch modes (v1.9.0): ``abupt`` (app/abupt_infer.py — anchor/query
transformer, validated eval recipe) and ``tripnet`` (app/tripnet_infer.py —
per-car triplane fit + shared U-Net/MLP head) branch out of ``predict_on_mesh``
before the standard sampling pipeline.
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree

from app import model_registry
from app.model_registry import REPO_ROOT

# Make the repo root and the transolver package importable before importing the
# proven script (it does a module-level `from data_loader import ...` and derives
# paths from `project_root`).
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "models", "transolver")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import single_car_inference_fixed as sci  # noqa: E402  (proven inference functions)

# The script sets project_root = os.getcwd() at import; pin it to the repo root so
# it works regardless of the server's working directory.
sci.project_root = REPO_ROOT


# ── Warm model cache (v1.8.0) ────────────────────────────────────────────────
# The original contract loaded the model on every request and freed it in a
# ``finally`` (low idle VRAM).  That load dominates per-request latency, so we
# now keep loaded models resident, keyed by name.  Repeat predictions skip the
# load entirely.  ``--workers 1`` (the launcher) makes this single in-process
# cache correct; ``clear_cache`` / ``unload_model`` give explicit VRAM control.
_MODEL_CACHE: Dict[str, Tuple] = {}


def _get_model(model_name: str, checkpoint: str, device: str) -> Tuple:
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    loaded = sci.load_wrapper_and_model(model_name, checkpoint, device)
    _MODEL_CACHE[model_name] = loaded
    return loaded


def prewarm(model_name: str, device: str = "cuda") -> None:
    """Load ``model_name`` into the cache ahead of the first request, and run one
    tiny dummy forward so cuDNN autotune / first-kernel compilation is paid at
    startup instead of on the first user prediction (otherwise ~40 s cold)."""
    meta = model_registry.get(model_name)
    if meta["mode"] not in ("tripnet", "abupt"):  # these load inside predict_on_mesh
        _get_model(model_name, meta["checkpoint"], device)
    try:  # warm the CUDA kernels at the real ~10k working size (cuDNN autotunes
        # per input shape, so a tiny sphere wouldn't warm the 10k-point path)
        predict_on_mesh(model_name, pv.Sphere(theta_resolution=170, phi_resolution=170),
                        device=device, strategy="decimate")
    except Exception as e:  # never block startup on the warm-forward
        print(f"[prewarm] dummy forward for {model_name} failed: {e}", flush=True)


def unload_model(model_name: str) -> bool:
    """Drop one cached model and free its VRAM.  Returns True if one was held."""
    dropped = _MODEL_CACHE.pop(model_name, None) is not None
    if model_name == "tripnet":
        from app import tripnet_infer
        dropped = tripnet_infer.clear() or dropped
    if dropped:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dropped


def clear_cache() -> None:
    """Drop all cached models and free VRAM."""
    _MODEL_CACHE.clear()
    try:
        from app import tripnet_infer
        tripnet_infer.clear()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cached_models() -> list:
    return sorted(_MODEL_CACHE.keys())


def _as_surface_polydata(mesh: pv.DataSet) -> pv.PolyData:
    """Return a triangulated PolyData surface (STL is already one; VTK may not be)."""
    if isinstance(mesh, pv.PolyData):
        return mesh
    return mesh.extract_surface()


def _fast_interpolate(mesh_points: np.ndarray, sample_points: np.ndarray,
                      values: np.ndarray, k: int = 6) -> np.ndarray:
    """k-NN inverse-distance interpolation from sampled predictions to the full mesh.

    Replaces the script's RBFInterpolator (a dense 10k×10k solve that took ~90 s on a
    500k-point car). KDTree k-NN IDW is ~1 s and visually equivalent for a dense
    surface sampling.
    """
    sample_points = np.asarray(sample_points, dtype=np.float64)
    mesh_points = np.asarray(mesh_points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    k = min(k, len(sample_points))
    tree = cKDTree(sample_points)
    dist, idx = tree.query(mesh_points, k=k)
    if k == 1:
        return values[idx].astype(np.float32)
    w = 1.0 / np.maximum(dist, 1e-8)
    # Where a mesh point coincides with a sample (dist≈0), use that sample exactly.
    exact = dist[:, 0] < 1e-8
    out = np.einsum("ij,ij->i", w, values[idx]) / w.sum(axis=1)
    out[exact] = values[idx[exact, 0]]
    return out.astype(np.float32)


def sample_surface_points(mesh: pv.PolyData, n_points: int = 10000,
                          seed: Optional[int] = 42) -> np.ndarray:
    """Sample up to ``n_points`` surface points (no ground-truth required)."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.asarray(mesh.points, dtype=np.float32)
    if len(pts) > n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    return pts


def _run_one(model, wrapper, config, coords: np.ndarray, surface: pv.PolyData,
             model_name: str, dispatch_mode: str, device: str) -> np.ndarray:
    """One forward pass: predict pressure at ``coords`` (a ~n_points subset of the
    surface).  The model trained on whole-car ~10k samples, so each call must pass
    a global subsample of that size to stay in-distribution."""
    if "transolver" in model_name.lower():
        preds = sci.run_inference_transolver(
            model, wrapper, coords, surface, config,
            ground_truth_pressures=None, device=device, model_name=model_name,
        )
    elif dispatch_mode in ("abupt", "tripnet"):
        # These modes branch out of predict_on_mesh before reaching here.
        raise RuntimeError(f"{dispatch_mode} is dispatched in predict_on_mesh, not _run_one.")
    else:
        preds = sci.run_inference_standard(
            model, wrapper, coords, ground_truth_pressures=None, device=device,
        )
    return np.asarray(preds, dtype=np.float32).flatten()[: len(coords)]


def _package(surface: pv.PolyData, full_preds: np.ndarray, metadata: Dict
             ) -> Tuple[pv.PolyData, Dict]:
    """Attach per-vertex pressures to a clean copy of ``surface`` + stats."""
    full_preds = np.asarray(full_preds, dtype=np.float32).flatten()
    out = surface.copy(deep=True)
    out.clear_data()
    out.point_data["pressure"] = full_preds
    metadata.update({
        "n_mesh_points": int(out.n_points),
        "pressure_min": float(np.min(full_preds)),
        "pressure_max": float(np.max(full_preds)),
        "pressure_mean": float(np.mean(full_preds)),
        "units": "Pa",
    })
    return out, metadata


def _predict_patches(model, wrapper, config, surface: pv.PolyData, model_name: str,
                     dispatch_mode: str, device: str, n_points: int, seed: int
                     ) -> Tuple[np.ndarray, int]:
    """Predict EVERY surface point directly (no interpolation) by running K passes
    over disjoint ~n_points global subsamples that together cover all points.

    Each pass is padded back up to ``n_points`` with already-seen points so the
    model always sees a full-size, in-distribution sample; only the not-yet-seen
    points in each pass are written.  K = ceil(N / n_points) passes.

    For transolver we compute the full-surface normals ONCE and index them per
    patch (``run_inference_transolver`` otherwise recomputes ``compute_normals`` +
    a full-mesh ``NearestNeighbors.fit`` on every call — ~K× the cost).  This is
    verified bit-for-bit identical to that function: the query points ARE surface
    points, so each point's nearest-neighbour normal is its own.
    """
    pts = np.asarray(surface.points)
    N = len(pts)
    is_transolver = "transolver" in model_name.lower()

    normals_full = None
    denorm = None
    if is_transolver:
        swn = surface.compute_normals(
            point_normals=True, auto_orient_normals=True, consistent_normals=True
        )
        normals_full = np.asarray(swn.point_data["Normals"], dtype=np.float32)
        _, denorm = wrapper.get_normalization_functions()

    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    full = np.zeros(N, dtype=np.float32)
    pos = 0
    n_passes = 0
    while pos < N:
        end = min(pos + n_points, N)
        new_idx = order[pos:end]
        chunk = new_idx
        if len(chunk) < n_points and pos > 0:  # pad the final short pass for context
            n_pad = min(n_points - len(chunk), pos)
            chunk = np.concatenate([new_idx, rng.choice(order[:pos], n_pad, replace=False)])

        if is_transolver:
            feats = np.concatenate(
                [pts[chunk].astype(np.float32), normals_full[chunk]], axis=1
            )
            ft = torch.tensor(feats, dtype=torch.float32)
            batch = (
                ft.unsqueeze(0),
                torch.zeros(ft.shape[0], dtype=torch.float32).unsqueeze(0),
            )
            data, _ = wrapper.preprocess_batch(batch)
            data = data.to(device)
            with torch.no_grad():
                pred = wrapper.forward(model, data)
            if denorm is not None:
                pred = denorm(pred)
            preds = pred.cpu().numpy().flatten()[: len(chunk)]
        else:
            preds = _run_one(
                model, wrapper, config, pts[chunk], surface,
                model_name, dispatch_mode, device,
            )
        full[new_idx] = preds[: len(new_idx)]
        pos = end
        n_passes += 1
    return full, n_passes


def predict_on_mesh(model_name: str, mesh: pv.DataSet, device: str = "cuda",
                    n_points: int = 10000, seed: int = 42,
                    strategy: str = "decimate") -> Tuple[pv.PolyData, Dict]:
    """Predict surface pressure on ``mesh`` using ``model_name`` (cached, warm).

    ``strategy``:
      * ``"decimate"`` (default, fast) — sample ~``n_points`` points, predict once,
        kNN-interpolate to the full mesh.  One forward pass.
      * ``"patch"`` (accurate) — cover all points with K=⌈N/n_points⌉ passes so
        every point is predicted directly (no interpolation blur).  K× slower.

    ``abupt`` and ``tripnet`` registry modes take their own validated paths
    (see app/abupt_infer.py and app/tripnet_infer.py).

    Returns (polydata_with_pressure, metadata).  Model is loaded once + kept
    resident (:func:`_get_model`); :func:`clear_cache` reclaims VRAM.
    """
    meta = model_registry.get(model_name)
    dispatch_mode = meta["mode"]
    checkpoint = meta["checkpoint"]
    surface = _as_surface_polydata(mesh)

    if dispatch_mode == "tripnet":
        from app import tripnet_infer  # lazy: open3d import only when used
        full_preds, extra = tripnet_infer.predict_full(
            surface, checkpoint, device=device, seed=seed,
        )
        return _package(surface, full_preds, {
            "model": model_name, "mode": dispatch_mode, "strategy": "triplane",
            "n_passes": 1, "n_surface_points": int(surface.n_points), **extra,
        })

    if dispatch_mode == "abupt":
        from app import abupt_infer
        cached = _MODEL_CACHE.get(model_name)
        if cached is None:  # own loader: EMA weights, weights_only=False (README)
            cached = (abupt_infer.load_model(checkpoint, device), None,
                      {"model_type": "abupt"})
            _MODEL_CACHE[model_name] = cached
        full_preds, extra = abupt_infer.predict_full(
            cached[0], surface, device=device, n_points=n_points, seed=seed,
            strategy=strategy, interpolate=_fast_interpolate,
        )
        return _package(surface, full_preds, {
            "model": model_name, "mode": dispatch_mode, **extra,
        })

    model, wrapper, config = _get_model(model_name, checkpoint, device)

    pts = np.asarray(surface.points)
    n_mesh = len(pts)

    if strategy == "patch" and n_mesh > n_points:
        full_preds, n_passes = _predict_patches(
            model, wrapper, config, surface, model_name, dispatch_mode, device,
            n_points, seed,
        )
        n_surface = n_mesh
    else:
        strategy = "decimate"
        sampled = sample_surface_points(surface, n_points=n_points, seed=seed)
        surface_preds = _run_one(
            model, wrapper, config, sampled, surface, model_name, dispatch_mode, device,
        )
        # Interpolate the sampled predictions onto the full surface (when sampled).
        full_preds = (
            _fast_interpolate(pts, sampled, surface_preds)
            if len(sampled) < n_mesh
            else surface_preds
        )
        n_surface = int(len(sampled))
        n_passes = 1

    return _package(surface, full_preds, {
        "model": model_name,
        "mode": dispatch_mode,
        "strategy": strategy,
        "n_passes": int(n_passes),
        "n_surface_points": int(n_surface),
    })
