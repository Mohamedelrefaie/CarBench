"""Auto-normalize an arbitrary car STL into the canonical DrivAerNet frame.

The pretrained models consume point coordinates as features, so an input STL must
be placed in the same frame as the training meshes or predictions degrade. The
canonical DrivAerNet frame (measured from the dataset) is:

    X = length  (~4.68 m, range ~[-0.81, 3.87])   streamwise
    Y = width   (~2.29 m, symmetric about 0)        lateral
    Z = height  (~1.35 m, ground plane at z = 0)    vertical
    units = meters,  length > width > height

`normalize_to_drivaernet` makes a best effort to align an arbitrary STL to this
frame: fix units (mm→m), reorder axes by extent (longest→X, …), uniformly scale so
the car length matches canonical, then place it (ground at 0, Y centered, X aligned).
Front/back and left/right orientation are ambiguous from geometry alone and can be
corrected with `axis_override` (flips) from the client.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyvista as pv

# Canonical DrivAerNet frame (from dataset; see app/stl_normalize docstring).
CANON_LENGTH = 4.681          # X extent (m)
CANON_X_CENTER = 1.5315       # (xmin+xmax)/2
CANON_Y_CENTER = 0.0          # symmetric
CANON_Z_MIN = 0.0             # ground plane


def normalize_to_drivaernet(
    mesh: pv.PolyData,
    axis_override: Optional[Dict] = None,
    scale_to_canonical: bool = True,
    frame_mode: Optional[str] = None,
) -> tuple[pv.PolyData, Dict]:
    """Return (aligned_mesh, report).

    axis_override (all optional):
      - "perm": explicit [i,j,k] mapping input axes -> (X,Y,Z). Overrides the
        extent-based guess.
      - "flip_x" / "flip_y" / "flip_z": bool, applied after permutation.
      - "assume_meters": bool, skip the mm→m unit heuristic.
    """
    axis_override = axis_override or {}
    out = mesh.copy(deep=True)
    pts = np.asarray(out.points, dtype=np.float64)
    report: Dict = {"input_bounds": [float(b) for b in out.bounds]}

    # 1) Units: a car in mm has extents ~thousands; in meters ~few.
    extent = pts.max(axis=0) - pts.min(axis=0)
    max_extent = float(extent.max())
    unit_scale = 1.0
    if not axis_override.get("assume_meters") and max_extent > 100.0:
        unit_scale = 0.001  # mm -> m
    pts *= unit_scale
    report["unit_scale"] = unit_scale

    # 2) Orientation: map longest extent -> X, middle -> Y, shortest -> Z.
    extent = pts.max(axis=0) - pts.min(axis=0)
    if "perm" in axis_override:
        perm = list(axis_override["perm"])
    else:
        perm = list(np.argsort(extent)[::-1])  # descending extent -> [long, mid, short]
    pts = pts[:, perm]
    report["axis_permutation"] = [int(p) for p in perm]

    # Optional flips (correct front/back, left/right, up/down ambiguity).
    for axis, key in ((0, "flip_x"), (1, "flip_y"), (2, "flip_z")):
        if axis_override.get(key):
            pts[:, axis] = -pts[:, axis]
    report["flips"] = {k: bool(axis_override.get(k)) for k in ("flip_x", "flip_y", "flip_z")}

    # Decide native vs canonical-fit. The model keys off ABSOLUTE coordinates,
    # so an in-distribution car (already real-meters, positioned in-frame) must
    # NOT be rescaled/shifted. 'auto' detects that; 'native'/'fit' force it.
    length0 = float(pts[:, 0].max() - pts[:, 0].min())
    xcenter0 = 0.5 * float(pts[:, 0].min() + pts[:, 0].max())
    zmin0 = float(pts[:, 2].min())
    _mode = (frame_mode or '').strip().lower()
    if _mode not in ('auto', 'native', 'fit'):
        _mode = 'fit' if scale_to_canonical else 'native'
    if _mode == 'auto':
        _in = (3.3 <= length0 <= 6.5) and (0.2 <= xcenter0 <= 2.9) and (-0.4 <= zmin0 <= 0.6)
        _mode = 'native' if _in else 'fit'
    do_fit = (_mode == 'fit')
    report['frame_mode'] = _mode

    # 3) Uniform scale so car length (X extent) matches canonical (preserves aspect).
    length = float(pts[:, 0].max() - pts[:, 0].min())
    geom_scale = 1.0
    if do_fit and length > 1e-6:
        geom_scale = CANON_LENGTH / length
        pts *= geom_scale
    report["geom_scale"] = geom_scale

    # 4) Placement: ground at z=0, Y centered at 0, X center aligned to canonical.
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, _ = pts.max(axis=0)
    if do_fit:
        pts[:, 0] += CANON_X_CENTER - 0.5 * (xmin + xmax)
    pts[:, 1] += CANON_Y_CENTER - 0.5 * (ymin + ymax)
    pts[:, 2] += CANON_Z_MIN - zmin

    out.points = pts.astype(np.float32)
    report["output_bounds"] = [float(b) for b in out.bounds]
    report["output_length_width_height"] = [
        float(out.bounds[1] - out.bounds[0]),
        float(out.bounds[3] - out.bounds[2]),
        float(out.bounds[5] - out.bounds[4]),
    ]
    # Sanity flag: warn if proportions are far from a typical car.
    l, w, h = report["output_length_width_height"]
    report["warnings"] = []
    if w > l or h > l:
        report["warnings"].append(
            "Width/height exceeds length after alignment — axis guess may be wrong; "
            "try an axis_override."
        )
    return out, report
