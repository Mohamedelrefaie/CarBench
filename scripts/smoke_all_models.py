#!/usr/bin/env python
"""Load every released checkpoint and predict on a few geometries.

A release gate: it catches a checkpoint that does not load, a wrapper whose
imports do not resolve outside the development tree, and predictions that are
not in physical units.

    python scripts/smoke_all_models.py --gpu 1 --n 2
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    assert a.gpu != "0", "GPU 0 is reserved"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    import numpy as np, pyvista as pv
    import carbench_paths as P
    from app import inference_core as IC
    from app.model_registry import MODELS

    cars = [l.strip() for l in open(P.require(P.TEST_IDS, "test split")) if l.strip()][: a.n]
    print(f"{len(MODELS)} models x {len(cars)} geometries\n")
    bad = 0
    for name in MODELS:
        if not os.path.exists(MODELS[name]["checkpoint"]):
            print(f"  {name:18} SKIP (weights not released; see checkpoints/README.md)")
            continue
        try:
            IC.prewarm(name, device="cuda")
            r2s = []
            for car in cars:
                mesh = IC._as_surface_polydata(pv.read(os.path.join(P.SURFACE_DIR, f"{car}.vtk")))
                key = next(k for k in ("p", "pressure", "Pressure", "P") if k in mesh.point_data)
                gt = np.asarray(mesh.point_data[key], float)
                out, _ = IC.predict_on_mesh(name, mesh, device="cuda", n_points=10000,
                                            seed=42, strategy="decimate")
                pr = np.asarray(out.point_data["pressure"], float)
                r2s.append(1 - np.sum((pr - gt) ** 2) / np.sum((gt - gt.mean()) ** 2))
            m = float(np.mean(r2s))
            ok = m > 0.5
            bad += (not ok)
            print(f"  {name:18} {'OK  ' if ok else 'SUSPECT'} mean R2 {m:.4f}")
        except Exception as e:
            bad += 1
            print(f"  {name:18} FAILED  {type(e).__name__}: {str(e)[:90]}")
    print("\nall models load and predict" if not bad else f"\n{bad} model(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
