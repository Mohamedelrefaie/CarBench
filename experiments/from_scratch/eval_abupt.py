#!/usr/bin/env python
"""Standalone, robust AB-UPT eval for the from-scratch comparison (rebuttal point 2).

Why standalone: the benchmark's comprehensive-summary path emits a ~0.5 GB per-point CSV and
an inf aggregate R^2 (reproducibly, for the PRETRAINED checkpoint too), and its fp16 forward
overflows on a few cars for a less-conditioned from-scratch model. This script evaluates a
checkpoint per-car in **fp32** using the same dense collator the wrapper uses (65536 geometry /
16384 supernodes / 16384 anchors + all remaining surface points as queries), denormalizes to
Pa, and computes the same metrics the table reports. Run it on BOTH checkpoints so the numbers
are apples-to-apples; the pretrained run doubles as a sanity check (should reproduce ~0.9675).

  python experiments/from_scratch/eval_abupt.py --checkpoint <ckpt.th> --tag from_scratch \
      --gpu 0 --out results_from_scratch/abupt/eval_<tag>.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ABUPT = os.path.join(REPO, "models", "abupt")
sys.path.insert(0, ABUPT)

from surface_only_model import SurfaceOnlyABUPT            # noqa: E402
from surface_abupt_collator import SurfaceOnlyAbuptCollator  # noqa: E402
from drivaernet_dataset import DrivAerNetDataset           # noqa: E402

PRESSURE_MEAN, PRESSURE_STD = -93.41, 120.79
TGT = {"surface_anchor_pressure", "surface_anchor_wallshearstress",
       "surface_query_pressure", "surface_query_wallshearstress"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="from_scratch")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-geometry-points", type=int, default=65536)
    ap.add_argument("--num-geometry-supernodes", type=int, default=16384)
    ap.add_argument("--num-surface-anchor-points", type=int, default=16384)
    ap.add_argument("--max-query-points", type=int, default=None,
                    help="truncate queries per car for speed (per-car metrics ~unchanged)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = "cuda"

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model_state_dict", ck))
    model = SurfaceOnlyABUPT().to(dev)
    model.load_state_dict(sd)
    model.eval()

    ds = DrivAerNetDataset(root=os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "PressureVTK", "Dataverse")),
                           split=args.split, skip_missing=True)
    col = SurfaceOnlyAbuptCollator(
        num_geometry_points=args.num_geometry_points,
        num_geometry_supernodes=args.num_geometry_supernodes,
        num_surface_anchor_points=args.num_surface_anchor_points,
        use_query_positions=True, dataset=ds)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[eval_abupt:{args.tag}] {n} {args.split} cars, fp32, "
          f"geo={args.num_geometry_points}/sn={args.num_geometry_supernodes}/"
          f"anc={args.num_surface_anchor_points}", flush=True)

    # accumulators replicating benchmark/metrics.py::MetricsCalculator EXACTLY (global mse/mae/
    # rmse/r2/max_error/percentiles; per-sample-averaged rel_l1/rel_l2).
    per_r2, per_mae, per_rl2, per_rl1 = [], [], [], []   # per-car (rel_* are CarBench metrics)
    sum_se = sum_ae = n_pts = 0.0
    sum_gt = sum_gt2 = 0.0          # for GLOBAL-mean SS_tot
    max_ae = 0.0
    abs_errs = []                  # pooled |error| for percentiles + max
    nonfinite = 0
    t0 = time.time()
    for i in range(n):
        s = {"surface_position_vtp": ds.getitem_surface_position_vtp(i),
             "surface_pressure": ds.getitem_surface_pressure(i),
             "surface_wallshearstress": ds.getitem_surface_wallshearstress(i)}
        b = col([s])
        if args.max_query_points and "surface_query_position" in b:
            q = b["surface_query_position"].shape[1]
            if q > args.max_query_points:
                sel = torch.randperm(q)[:args.max_query_points]
                b["surface_query_position"] = b["surface_query_position"][:, sel]
                b["surface_query_pressure"] = b["surface_query_pressure"][sel]
        inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items() if k not in TGT}
        with torch.no_grad():                    # fp32 (no autocast) -> no fp16 overflow
            out = model(**inp)
        pred = torch.cat([out["surface_anchor_pressure"], out["surface_query_pressure"]]).float().squeeze(-1).cpu().numpy()
        gt = torch.cat([b["surface_anchor_pressure"], b["surface_query_pressure"]]).float().squeeze(-1).numpy()
        pred = pred * PRESSURE_STD + PRESSURE_MEAN
        gt = gt * PRESSURE_STD + PRESSURE_MEAN
        if not np.isfinite(pred).all():
            nonfinite += 1
            pred = np.nan_to_num(pred, nan=gt.mean(), posinf=gt.max(), neginf=gt.min())
        d = pred - gt
        ad = np.abs(d)
        sum_se += float((d * d).sum()); sum_ae += float(ad.sum()); n_pts += d.size
        sum_gt += float(gt.sum()); sum_gt2 += float((gt * gt).sum())
        max_ae = max(max_ae, float(ad.max()))
        abs_errs.append(ad.astype(np.float32))
        per_rl2.append(float(np.linalg.norm(d) / (np.linalg.norm(gt) + 1e-12)))
        per_rl1.append(float(ad.sum() / (np.abs(gt).sum() + 1e-12)))
        per_r2.append(1 - float((d ** 2).sum()) / float(((gt - gt.mean()) ** 2).sum() + 1e-12))
        per_mae.append(float(ad.mean()))
        if i % 100 == 0:
            print(f"  {i+1}/{n} per-car R2={per_r2[-1]:.4f} {(time.time()-t0)/(i+1):.2f}s/car", flush=True)

    gmean = sum_gt / n_pts
    ss_tot = sum_gt2 - n_pts * gmean * gmean           # GLOBAL-mean SS_tot (matches MetricsCalculator)
    all_ad = np.concatenate(abs_errs)
    p50, p90, p95, p99 = (float(x) for x in np.percentile(all_ad, [50, 90, 95, 99]))
    res = {
        "tag": args.tag, "checkpoint": args.checkpoint, "n_cars": n, "nonfinite_cars": nonfinite,
        "units": "m2/s2_kinematic",
        # ---- CarBench / benchmark.metrics.MetricsCalculator metric set ----
        "mse": sum_se / n_pts,
        "mae": sum_ae / n_pts,
        "rmse": float(np.sqrt(sum_se / n_pts)),
        "r2": 1 - sum_se / ss_tot,                     # GLOBAL (pooled, global mean)
        "max_error": max_ae,
        "rel_l1": float(np.mean(per_rl1)),             # per-car ratio, averaged
        "rel_l2": float(np.mean(per_rl2)),
        "p50_error": p50, "p90_error": p90, "p95_error": p95, "p99_error": p99,
        # ---- extras (not CarBench): per-car arrays for paired bootstrap ----
        "r2_mean_percar": float(np.mean(per_r2)),
        "per_car_r2": per_r2, "per_car_mae": per_mae,
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n[eval_abupt:{args.tag}] (CarBench metrics, m2/s2) "
          f"MSE={res['mse']:.1f} MAE={res['mae']:.2f} RMSE={res['rmse']:.2f} R2={res['r2']:.4f} "
          f"relL1={res['rel_l1']:.4f} relL2={res['rel_l2']:.4f} max={res['max_error']:.1f} "
          f"P50={p50:.2f} P90={p90:.2f} P95={p95:.2f} P99={p99:.2f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
