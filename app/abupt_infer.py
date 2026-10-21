"""AB-UPT inference glue (registry mode ``"abupt"``).

Follows the validated evaluation path (models/surface-abupt-drivaernet/
evaluate_full_test_set_updated.py) but builds the batch directly so the
anchor/query index mapping stays under our control (the stock collator
randperms internally and discards the permutation):

* positions are isotropically rescaled from the DrivAerNet meter frame into
  [0, 1000] with the dataset-wide scalar bounds (drivaernet_stats),
* 65536-point geometry cloud + 16384 supernodes + 16384 anchors,
* queries run in chunks of 16384 per forward (the validated shape) under
  fp16 autocast,
* predictions are ALWAYS denormalized with the dataset pressure moments —
  the sci helper's heuristic denorm is GT-dependent and wrong for GT-free
  serving.

The model instance comes from ``sci.load_abupt_model`` via the inference
core's warm cache; this module needs no imports from models/.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Tuple

import numpy as np
import pyvista as pv
import torch

from app.model_registry import REPO_ROOT

# Dataset-wide constants (models/abupt/drivaernet_dataset.py, drivaernet_stats)
_RAW_POS_MIN = -1.19442
_RAW_POS_MAX = 4.20453
_P_MEAN = -93.4098
_P_STD = 120.787
_SCALE = 1000.0

# Validated sampling envelope (models/abupt/config.yaml + eval scripts)
_N_GEOMETRY = 65536
_N_SUPERNODES = 16384
_N_ANCHORS = 16384
_N_QUERY_CHUNK = 16384


def load_model(checkpoint_path: str, device: str):
    """Build ``SurfaceOnlyABUPT`` and load the (EMA) weights.

    Per README_TRIPNET_ABUPT.md: import from models/abupt/surface_only_model.py
    (models/abupt/model.py is a stub), prefer the checkpoint's EMA ``state_dict``,
    and use ``weights_only=False`` (the from-scratch dicts contain numpy/plain
    dicts that weights_only=True rejects).
    """
    abupt_dir = os.path.join(REPO_ROOT, "models", "abupt")
    if abupt_dir not in sys.path:
        sys.path.insert(0, abupt_dir)
    from surface_only_model import SurfaceOnlyABUPT

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck
    model = SurfaceOnlyABUPT()  # defaults == ck["ctor_kwargs"] (minus 'kind')
    model.load_state_dict(sd)
    return model.eval().to(device)


def _to_model_frame(pts: np.ndarray) -> torch.Tensor:
    """DrivAerNet meters -> the model's isotropic [0, 1000] space."""
    span = _RAW_POS_MAX - _RAW_POS_MIN
    return torch.from_numpy(((pts - _RAW_POS_MIN) / span * _SCALE).astype(np.float32))


def predict_full(model, surface: pv.PolyData, device: str, n_points: int,
                 seed: int, strategy: str,
                 interpolate: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                 ) -> Tuple[np.ndarray, Dict]:
    """Pressure at every vertex of ``surface`` (DrivAerNet meter frame).

    Every vertex is predicted directly, chunked at the validated 16384 queries
    per forward (anchors/geometry fixed across chunks) — measured ~2 s warm on
    a 525k-point car with R² matching the from-scratch eval, so the sampled+
    interpolated path other models use for speed is never worth it here.
    ``strategy``/``n_points``/``interpolate`` are accepted for signature parity
    and ignored.
    """
    pts = np.asarray(surface.points, dtype=np.float32)
    n = len(pts)
    gen = torch.Generator().manual_seed(seed)
    pos = _to_model_frame(pts)  # (N, 3) in [0, 1000]

    # Geometry cloud + supernodes (the encoder's view of the car).
    n_geo = min(_N_GEOMETRY, n)
    geo_idx = torch.randperm(n, generator=gen)[:n_geo]
    supernode_idx = torch.randperm(n_geo, generator=gen)[: min(_N_SUPERNODES, n_geo)]

    # Anchors: a global random subset providing the decoder's context tokens.
    anchor_idx = torch.randperm(n, generator=gen)[: min(_N_ANCHORS, n)]

    query_idx = np.arange(n)
    used_strategy = "direct"

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    geometry_position = pos[geo_idx].to(device)
    supernode_idx_dev = supernode_idx.to(device)
    anchor_position = pos[anchor_idx].unsqueeze(0).to(device)

    model.eval()
    preds = np.empty(len(query_idx), dtype=np.float32)
    n_passes = 0
    with torch.no_grad():
        for s in range(0, len(query_idx), _N_QUERY_CHUNK):
            chunk = query_idx[s : s + _N_QUERY_CHUNK]
            batch = dict(
                geometry_position=geometry_position,
                geometry_supernode_idx=supernode_idx_dev,
                geometry_batch_idx=None,
                surface_anchor_position=anchor_position,
                surface_query_position=pos[chunk].unsqueeze(0).to(device),
            )
            with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                dtype=torch.float16, enabled=use_cuda):
                out = model(**batch)
            preds[s : s + len(chunk)] = (
                out["surface_query_pressure"].float().cpu().numpy().flatten()[: len(chunk)]
            )
            n_passes += 1

    preds = preds * _P_STD + _P_MEAN  # explicit denorm, never heuristic
    full = preds  # query_idx is arange(n): already in vertex order

    extra = {
        "strategy": used_strategy,
        "n_passes": int(n_passes),
        "n_surface_points": int(len(query_idx)),
        "n_anchors": int(len(anchor_idx)),
        "n_geometry_points": int(n_geo),
    }
    return full, extra
