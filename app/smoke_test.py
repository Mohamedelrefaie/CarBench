"""Headless smoke test for the inference app (run on the cluster).

Validates, without the HTTP layer or GUI:
  1. imports + model registry
  2. STL auto-normalization (incl. a mm-scaled, rotated copy -> canonical frame)
  3. end-to-end prediction on a known DrivAerNet car, comparing predicted vs the
     ground-truth `p` field (MAE should be ~benchmark level, not garbage)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m app.smoke_test \
        --vtk $CARBENCH_SURFACE_DIR/E_S_WWC_WM_005.vtk \
        --model transolver
"""
from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from app import model_registry
from app.inference_core import predict_on_mesh
from app.stl_normalize import normalize_to_drivaernet


def test_normalize(vtk_path: str):
    print("\n[normalize] mm-scaled + axis-swapped copy -> canonical frame")
    m = pv.read(vtk_path)
    surf = m.extract_surface() if not isinstance(m, pv.PolyData) else m
    # Simulate a 'bad' STL: millimeters + axes permuted (length on Z).
    bad = surf.copy(deep=True)
    pts = np.asarray(bad.points)[:, [2, 1, 0]] * 1000.0  # swap X<->Z, to mm
    bad.points = pts
    aligned, report = normalize_to_drivaernet(bad)
    l, w, h = report["output_length_width_height"]
    print(f"   input bounds : {[round(b,1) for b in bad.bounds]}")
    print(f"   unit_scale={report['unit_scale']}  perm={report['axis_permutation']}  geom_scale={report['geom_scale']:.4f}")
    print(f"   output L/W/H : {l:.3f} / {w:.3f} / {h:.3f}  (canonical ~4.68/2.29/1.35)")
    assert abs(l - 4.681) < 0.2, "length not aligned to canonical"
    assert l >= w >= h, "axis ordering wrong"
    assert abs(aligned.bounds[4]) < 1e-3, "ground plane (min z) not at 0"
    print("   OK")


def test_predict(vtk_path: str, model: str):
    print(f"\n[predict] {model} on {vtk_path.split('/')[-1]}")
    mesh = pv.read(vtk_path)
    surf = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
    gt = np.asarray(surf.point_data["p"], dtype=np.float32)

    out, meta = predict_on_mesh(model, surf, device="cuda")
    pred = np.asarray(out.point_data["pressure"], dtype=np.float32)

    # Align predicted to GT points by nearest neighbour (robust to any reordering).
    tree = cKDTree(np.asarray(surf.points))
    _, idx = tree.query(np.asarray(out.points))
    mae = float(np.mean(np.abs(pred - gt[idx])))
    print(f"   pressure range pred: [{meta['pressure_min']:.1f}, {meta['pressure_max']:.1f}] Pa")
    print(f"   ground-truth range : [{gt.min():.1f}, {gt.max():.1f}] Pa")
    print(f"   full-surface MAE vs GT: {mae:.2f} Pa   (benchmark transolver ~12.7)")
    assert mae < 40.0, f"MAE {mae:.1f} too high — inference path likely broken"
    print("   OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", required=True, help="A DrivAerNet VTK with a 'p' field")
    ap.add_argument("--model", default="transolver")
    args = ap.parse_args()

    print("[registry] available models:")
    for m in model_registry.list_models():
        print(f"   - {m['name']:<18} {m['mode']:<10} {m['description']}")

    test_normalize(args.vtk)
    test_predict(args.vtk, args.model)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
