#!/usr/bin/env python
"""Train + evaluate the TripNet pressure predictor FROM SCRATCH (rebuttal point 2).

Uses per-car triplanes fitted by models/tripnet/occupancy_fit.py (stage1/stage2). Trains the
shared U-Net + 4-layer MLP on surface pressure (normalized MSE), then evaluates on the test
cars, reporting BOTH:
  - normalized metrics (relL2, relL1, MSE, MAE) -> compare to the TripNet paper's reported
    pretrained numbers (arXiv:2503.17400: relL2 0.2035, relL1 0.1852, MSE 0.0423, MAE 0.1110)
  - Pascal metrics (R2, MAE, RMSE, relL2) -> line up with our benchmark table

Usage:
  python experiments/from_scratch/train_tripnet.py \
     --tri-dir <triplane_dir> --train-cars <ids> --test-cars <ids> \
     --config models/tripnet/config.yaml --save-dir results_from_scratch/tripnet \
     --gpu 0 [--epochs N --batch-size 8 --smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TN_DIR = os.path.join(REPO_ROOT, "models", "tripnet")
sys.path.insert(0, TN_DIR)

from model import TripNet                                  # noqa: E402
from data_loader import (TriplanePressureDataset, collate,  # noqa: E402
                         PRESSURE_MEAN, PRESSURE_STD, _read_vtk_surface)
from triplane import sample_triplane                        # noqa: E402


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()]


@torch.no_grad()
def predict_car(model, tri_dir, dataset_path, cid, device, chunk=200_000):
    """Predict normalized pressure at ALL surface points of one car."""
    tp = torch.load(os.path.join(tri_dir, f"{cid}.pt"), map_location="cpu", weights_only=False)
    planes = tp["planes"].float().to(device)                  # (1,3,C,H,W) or (3,C,H,W)
    if planes.dim() == 4:
        planes = planes.unsqueeze(0)
    lo = np.asarray(tp["bbox_min"], np.float32); hi = np.asarray(tp["bbox_max"], np.float32)
    pts, p = _read_vtk_surface(os.path.join(dataset_path, f"{cid}.vtk"))
    center = (lo + hi) * 0.5; scale = np.clip((hi - lo) * 0.5, 1e-6, None)
    ndc = torch.from_numpy(((pts - center) / scale).astype(np.float32)).to(device)
    refined = model.refine(planes)                            # (1,3,C,H,W)
    preds = []
    for i in range(0, ndc.shape[0], chunk):
        feat = sample_triplane(refined, ndc[i:i + chunk].unsqueeze(0))
        preds.append(model.mlp(feat).squeeze(0).cpu())
    pred_norm = torch.cat(preds).numpy()
    gt_norm = (p - PRESSURE_MEAN) / PRESSURE_STD
    return pred_norm, gt_norm


def evaluate(model, tri_dir, dataset_path, cids, device):
    """Full CarBench metric set (== benchmark/metrics.py::MetricsCalculator) on denormalized
    kinematic pressure (m^2/s^2); normalized-pressure metrics kept for the paper cross-ref."""
    model.eval()
    # denormalized (m^2/s^2) accumulators, matching MetricsCalculator exactly
    sum_se = sum_ae = n_pts = 0.0
    sum_gt = sum_gt2 = 0.0
    max_ae = 0.0
    abs_errs = []
    rl2, rl1 = [], []
    # normalized accumulators (paper convention)
    sse_n = sae_n = n_n = 0.0
    rl2n, rl1n = [], []
    used = 0
    for cid in cids:
        if not os.path.exists(os.path.join(tri_dir, f"{cid}.pt")):
            continue
        pred_n, gt_n = predict_car(model, tri_dir, dataset_path, cid, device)  # normalized
        pred = pred_n * PRESSURE_STD + PRESSURE_MEAN
        gt = gt_n * PRESSURE_STD + PRESSURE_MEAN
        d = pred - gt; ad = np.abs(d)
        sum_se += float((d * d).sum()); sum_ae += float(ad.sum()); n_pts += d.size
        sum_gt += float(gt.sum()); sum_gt2 += float((gt * gt).sum())
        max_ae = max(max_ae, float(ad.max())); abs_errs.append(ad.astype(np.float32))
        rl2.append(float(np.linalg.norm(d) / (np.linalg.norm(gt) + 1e-9)))
        rl1.append(float(ad.sum() / (np.abs(gt).sum() + 1e-9)))
        dn = pred_n - gt_n
        sse_n += float((dn * dn).sum()); sae_n += float(np.abs(dn).sum()); n_n += dn.size
        rl2n.append(float(np.linalg.norm(dn) / (np.linalg.norm(gt_n) + 1e-9)))
        rl1n.append(float(np.abs(dn).sum() / (np.abs(gt_n).sum() + 1e-9)))
        used += 1
    gmean = sum_gt / n_pts
    ss_tot = sum_gt2 - n_pts * gmean * gmean
    p50, p90, p95, p99 = (float(x) for x in np.percentile(np.concatenate(abs_errs), [50, 90, 95, 99]))
    return {
        "units": "m2/s2_kinematic", "n_cars": used,
        # ---- CarBench metric set (denormalized, m^2/s^2) ----
        "mse": sum_se / n_pts, "mae": sum_ae / n_pts, "rmse": float(np.sqrt(sum_se / n_pts)),
        "r2": 1 - sum_se / ss_tot, "max_error": max_ae,
        "rel_l1": float(np.mean(rl1)), "rel_l2": float(np.mean(rl2)),
        "p50_error": p50, "p90_error": p90, "p95_error": p95, "p99_error": p99,
        # ---- normalized (TripNet-paper convention, for cross-reference) ----
        "rel_l2_norm": float(np.mean(rl2n)), "rel_l1_norm": float(np.mean(rl1n)),
        "mse_norm": sse_n / n_n, "mae_norm": sae_n / n_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tri-dir", required=True)
    ap.add_argument("--dataset-path", default=os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "PressureVTK", "Dataverse")))
    ap.add_argument("--train-cars", required=True)
    ap.add_argument("--test-cars", required=True)
    ap.add_argument("--config", default=os.path.join(TN_DIR, "config.yaml"))
    ap.add_argument("--save-dir", default=os.path.join(REPO_ROOT, "results_from_scratch", "tripnet"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--n-query", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="load saved predictor checkpoint and only re-evaluate (full metric set)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = yaml.safe_load(open(args.config))
    epochs = args.epochs or (2 if args.smoke else cfg["epochs"])
    bs = args.batch_size or cfg["batch_size"]
    nq = args.n_query or cfg["n_query"]
    seed = int(cfg.get("seed", 42)); torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda"

    train_cars = read_ids(args.train_cars); test_cars = read_ids(args.test_cars)
    ds = TriplanePressureDataset(args.tri_dir, args.dataset_path, train_cars, n_query=nq, train=True, seed=seed)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=len(ds) > bs, persistent_workers=args.workers > 0)
    print(f"[tripnet] train_cars={len(ds.cids)} test_cars={len(test_cars)} epochs={epochs} bs={bs} nq={nq}", flush=True)

    model = TripNet(channels=cfg["channels"], res=cfg["res"],
                    unet_base=cfg["unet_base"], mlp_width=cfg["mlp_width"]).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[tripnet] predictor params={nparam:,}", flush=True)

    ckpt_path = os.path.join(args.save_dir, "tripnet_from_scratch.pt")
    hist = []
    t0 = time.time()
    if args.eval_only:
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"]); hist = sd.get("train_history", [])
        print(f"[tripnet] eval-only: loaded {ckpt_path}", flush=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        for ep in range(1, epochs + 1):
            model.train(); tot = 0.0; nb = 0
            for batch in loader:
                planes = batch["planes"].to(device)
                q = batch["query_ndc"].to(device); tgt = batch["pressure"].to(device)
                pred = model(planes, q)
                loss = F.mse_loss(pred, tgt)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            hist.append(tot / max(nb, 1))
            print(f"  epoch {ep}/{epochs} mse {hist[-1]:.5f} {(time.time()-t0)/60:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "config": cfg, "nparam": nparam,
                    "train_history": hist}, ckpt_path)

    print("[tripnet] evaluating on test cars (full CarBench metric set)...", flush=True)
    metrics = evaluate(model, args.tri_dir, args.dataset_path, test_cars, device)
    metrics.update({"params": nparam, "epochs": epochs, "train_cars": len(ds.cids),
                    "train_minutes": (time.time() - t0) / 60})
    json.dump(metrics, open(os.path.join(args.save_dir, "tripnet_metrics.json"), "w"), indent=2)
    print("\n=== TripNet from-scratch (test) ===")
    print(f"  CarBench (m2/s2): MSE {metrics['mse']:.1f} | MAE {metrics['mae']:.2f} | "
          f"RMSE {metrics['rmse']:.2f} | R2 {metrics['r2']:.4f} | relL1 {metrics['rel_l1']:.4f} | "
          f"relL2 {metrics['rel_l2']:.4f} | max {metrics['max_error']:.1f}")
    print(f"  percentiles |err|: P50 {metrics['p50_error']:.2f} P90 {metrics['p90_error']:.2f} "
          f"P95 {metrics['p95_error']:.2f} P99 {metrics['p99_error']:.2f}")
    print(f"  (normalized, paper ref: relL2 {metrics['rel_l2_norm']:.4f} relL1 {metrics['rel_l1_norm']:.4f} "
          f"MSE {metrics['mse_norm']:.4f} MAE {metrics['mae_norm']:.4f}; paper 0.2035/0.1852/0.0423/0.1110)")
    print(f"  n={metrics['n_cars']} -> {args.save_dir}/tripnet_metrics.json", flush=True)


if __name__ == "__main__":
    main()
