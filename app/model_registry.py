"""Registry of pretrained surface-pressure models exposed by the inference app.

Maps a model name (which must match a ``models/<name>/`` folder so the wrapper
loader can find it) to its checkpoint and the inference dispatch mode. Checkpoints are the released weights in ``checkpoints/``; each reproduces the
row named in ``checkpoints/MANIFEST.json``.

Dispatch modes:
  - ``transolver`` : 6D surface features [x,y,z,nx,ny,nz] (Transolver family)
  - ``standard``   : raw point cloud (PointNet/RegDGCNN/PointTransformer/PointMAE/FNO)
  - ``abupt``      : anchor/query transformer (app/abupt_infer.py; sci loads it)
  - ``tripnet``    : per-car triplane fit + shared U-Net/MLP head
                     (app/tripnet_infer.py; ~20 s geometry fit on first use,
                     cached per mesh afterwards)
"""
from __future__ import annotations

import os
from typing import Dict, List

# Repo root = parent of this app/ directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CHECKPOINT_DIR = os.environ.get(
    "CARBENCH_CHECKPOINT_DIR", os.path.join(REPO_ROOT, "checkpoints"))


def _ckpt(name: str) -> str:
    """Resolve a released checkpoint by model name.

    Weights are distributed separately from the code (see checkpoints/README.md);
    ``scripts/download_checkpoints.py`` fetches them and verifies each SHA-256
    against checkpoints/MANIFEST.json.
    """
    for ext in (".pth", ".pt", ".th"):
        p = os.path.join(CHECKPOINT_DIR, name + ext)
        if os.path.exists(p):
            return p
    return os.path.join(CHECKPOINT_DIR, name + ".pth")


# name -> metadata. `name` MUST match models/<name>/ for wrapper discovery.
MODELS: Dict[str, Dict] = {
    "transolver": {
        "mode": "transolver",
        "checkpoint": _ckpt("transolver"),
        "description": "Transolver — physics-informed transformer (best accuracy/size balance)",
    },
    "transolver_large": {
        "mode": "transolver",
        "checkpoint": _ckpt("transolver_large"),
        "description": "Transolver-Large — highest-accuracy surface model (R² 0.96)",
    },
    "transolver_plus": {
        "mode": "transolver",
        "checkpoint": _ckpt("transolver_plus"),
        "description": "Transolver++ — efficient surface transformer",
    },
    "regdgcnn": {
        "mode": "standard",
        "checkpoint": _ckpt("regdgcnn"),
        "description": "RegDGCNN — dynamic graph CNN",
    },
    "point_transformer": {
        "mode": "standard",
        "checkpoint": _ckpt("point_transformer"),
        "description": "Point Transformer — attention over point clouds",
    },
    "pointnet": {
        "mode": "standard",
        "checkpoint": _ckpt("pointnet"),
        "description": "PointNet — baseline point-cloud net",
    },
    "pointnet_large": {
        "mode": "standard",
        "checkpoint": _ckpt("pointnet_large"),
        "released": False,   # weights not distributed; retrain with scripts/run_single_model.py
        "description": "PointNet-Large — wider PointNet",
    },
    "pointmae": {
        "mode": "standard",
        "checkpoint": _ckpt("pointmae"),
        "description": "PointMAE — masked-autoencoder encoder",
    },
    "fno": {
        "mode": "standard",
        "checkpoint": _ckpt("fno"),
        "description": "FNO — Fourier neural operator",
    },
    # From-scratch reimplementations (README_TRIPNET_ABUPT.md) — NOT the
    # vendored pretrained copies.
    "abupt": {
        "mode": "abupt",
        "checkpoint": _ckpt("abupt"),
        "description": "AB-UPT — anchor-branched universal physics transformer (R² 0.96)",
    },
    "tripnet": {
        "mode": "tripnet",
        "checkpoint": _ckpt("tripnet"),
        "description": "TripNet — triplane neural field, fits car geometry on first use (R² 0.95)",
    },
}


def list_models(only_available: bool = True) -> List[Dict]:
    """Return [{name, description, mode, available}] for the dropdown."""
    out = []
    for name, meta in MODELS.items():
        available = os.path.exists(meta["checkpoint"])
        if only_available and not available:
            continue
        out.append({
            "name": name,
            "description": meta["description"],
            "mode": meta["mode"],
            "available": available,
        })
    return out


def get(name: str) -> Dict:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODELS)}")
    return MODELS[name]
