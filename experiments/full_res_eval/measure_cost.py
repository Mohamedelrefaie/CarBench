#!/usr/bin/env python
"""Wall-clock cost of each full-resolution evaluation protocol (reviewer R1.1).

Patchwise evaluation buys accuracy at a price: it needs ceil(N/n_points) forward
passes per geometry instead of one. This script measures that price per model on
a quiet GPU, separating the three protocols:

    t_sub10k  : one forward pass on a 10,000-point subsample
    t_interp  : t_sub10k + upscaling to all mesh nodes (kNN-IDW or RBF)
    t_patch   : ceil(N/10k) forward passes covering every node

Mesh loading is excluded (identical for all protocols); CUDA is synchronised
around every timed region.

    python measure_cost.py --gpu 1 --n-cars 20
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SURF_DIR = os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "PressureVTK", "Dataverse"))
TEST_IDS = os.path.join(os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "train_val_test_splits")), "test_design_ids.txt")
MODELS = ["pointnet", "fno", "pointmae", "pointnet_large", "regdgcnn",
          "point_transformer", "tripnet", "transolver_plus", "transolver",
          "transolver_large", "abupt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-cars", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--with-rbf", action="store_true",
                    help="also time the RBF upscaling (~90 s/car)")
    ap.add_argument("--out", default=os.path.join(HERE, "cost_table.json"))
    args = ap.parse_args()
    assert args.gpu != "0", "GPU 0 is reserved"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import pyvista as pv
    import torch
    from scipy.interpolate import RBFInterpolator

    from app import inference_core as IC
    from run_full_res_eval import predict_at

    cars = [l.strip() for l in open(TEST_IDS) if l.strip()][: args.n_cars]
    meshes = []
    for c in cars:
        p = os.path.join(SURF_DIR, f"{c}.vtk")
        if os.path.exists(p):
            meshes.append(IC._as_surface_polydata(pv.read(p)))
    print(f"timing on {len(meshes)} geometries "
          f"({meshes[0].n_points:,} nodes each, ~{args.n_points} pts/pass)", flush=True)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    out = {}
    for m in args.models:
        try:
            IC.prewarm(m, device="cuda")
        except Exception as e:
            print(f"  !! {m}: prewarm failed ({e})")
            continue
        t_dec, t_patch, npass = [], [], []
        for mesh in meshes:
            # Interpolated protocol, timed exactly as run_full_res_eval performs it:
            # a 10k prediction followed by upscaling. Routing this through
            # predict_on_mesh(strategy="decimate") would time a *direct* full-mesh
            # prediction for the query-based models, which is a different protocol.
            sync(); t0 = time.perf_counter()
            np.random.seed(42)
            idx = (np.random.choice(mesh.n_points, args.n_points, replace=False)
                   if mesh.n_points > args.n_points else np.arange(mesh.n_points))
            p10k = predict_at(m, mesh, idx, args.n_points, 42, IC, pv)
            IC._fast_interpolate(np.asarray(mesh.points, dtype=np.float64),
                                 np.asarray(mesh.points, dtype=np.float64)[idx], p10k)
            sync(); t_dec.append(time.perf_counter() - t0)

            sync(); t0 = time.perf_counter()
            _, meta_p = IC.predict_on_mesh(m, mesh, device="cuda",
                                           n_points=args.n_points, strategy="patch")
            sync(); t_patch.append(time.perf_counter() - t0)
            npass.append(int(meta_p.get("n_passes", 0)))

        rec = {"t_interp_knn_s": float(np.median(t_dec)),
               "t_patch_s": float(np.median(t_patch)),
               "n_passes": int(np.median(npass)) if npass else 0,
               "ratio_patch_over_interp": float(np.median(t_patch) / max(np.median(t_dec), 1e-9)),
               "n_cars_timed": len(meshes)}

        if args.with_rbf:
            mesh = meshes[0]
            pts = np.asarray(mesh.points)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), args.n_points, replace=False)
            vals = np.asarray(mesh.point_data.get("p", np.zeros(len(pts))))[idx]
            t0 = time.perf_counter()
            RBFInterpolator(pts[idx], vals, kernel="linear", epsilon=2)(pts)
            rec["t_rbf_upscale_only_s"] = time.perf_counter() - t0

        out[m] = rec
        print(f"  {m:18s} interp {rec['t_interp_knn_s']:7.3f}s | "
              f"patch {rec['t_patch_s']:8.3f}s ({rec['n_passes']} passes) | "
              f"x{rec['ratio_patch_over_interp']:.1f}", flush=True)
        IC.clear_cache()

    json.dump({"n_points": args.n_points, "per_model": out}, open(args.out, "w"), indent=2)

    ratios = [v["ratio_patch_over_interp"] for v in out.values()]
    tot_i = sum(v["t_interp_knn_s"] for v in out.values()) * 1154 / 3600
    tot_p = sum(v["t_patch_s"] for v in out.values()) * 1154 / 3600
    print(f"\nmedian slowdown x{np.median(ratios):.1f} "
          f"(range x{min(ratios):.1f}-x{max(ratios):.1f})")
    print(f"extrapolated to 11 models x 1,154 geometries: "
          f"interpolation {tot_i:.1f} GPU-h vs patchwise {tot_p:.1f} GPU-h")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
