#!/usr/bin/env python3
"""Paired bootstrap: from-scratch vs pretrained AB-UPT, over the test cars.

Reads per-car metrics (aligned by test-split index) from the two eval JSONs produced by
eval_abupt.py and bootstraps the paired difference (pretrained - from_scratch). Reports the
mean gap and 95% CI for R2 and MAE so the rebuttal can state whether the pretrained advantage
is statistically meaningful.

  python experiments/from_scratch/paired_bootstrap_abupt.py \
     [--pre eval_pretrained_fast.json] [--fs eval_from_scratch_fast.json] [-B 10000]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
AB = os.path.join(REPO, "results_from_scratch", "abupt")


def boot(diff, B, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = diff[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=os.path.join(AB, "eval_pretrained_fast.json"))
    ap.add_argument("--fs", default=os.path.join(AB, "eval_from_scratch_fast.json"))
    ap.add_argument("-B", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(HERE, "abupt_paired_bootstrap.json"))
    args = ap.parse_args()

    pre = json.load(open(args.pre)); fs = json.load(open(args.fs))
    pr2 = np.array(pre["per_car_r2"]); fr2 = np.array(fs["per_car_r2"])
    pma = np.array(pre["per_car_mae"]); fma = np.array(fs["per_car_mae"])
    assert len(pr2) == len(fr2), f"car count mismatch {len(pr2)} vs {len(fr2)}"
    n = len(pr2)

    # gap = pretrained - from_scratch (positive => pretrained better)
    dr2 = pr2 - fr2
    dma = fma - pma   # MAE: from_scratch - pretrained (positive => from_scratch worse)
    m_r2, lo_r2, hi_r2 = boot(dr2, args.B)
    m_ma, lo_ma, hi_ma = boot(dma, args.B)

    res = {
        "n_cars": n, "B": args.B,
        "pretrained_R2_mean": float(pr2.mean()), "from_scratch_R2_mean": float(fr2.mean()),
        "deltaR2_pretrained_minus_fromscratch": {"mean": m_r2, "ci95": [lo_r2, hi_r2]},
        "pretrained_MAE_mean": float(pma.mean()), "from_scratch_MAE_mean": float(fma.mean()),
        "deltaMAE_fromscratch_minus_pretrained_pa": {"mean": m_ma, "ci95": [lo_ma, hi_ma]},
        "win_rate_fromscratch_better_R2": float((fr2 > pr2).mean()),
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"AB-UPT paired bootstrap (n={n} cars, B={args.B})")
    print(f"  pretrained   mean per-car R2 = {pr2.mean():.4f}  MAE = {pma.mean():.2f} Pa")
    print(f"  from-scratch mean per-car R2 = {fr2.mean():.4f}  MAE = {fma.mean():.2f} Pa")
    print(f"  ΔR2  (pre - fs)        = {m_r2:+.4f}  95% CI [{lo_r2:+.4f}, {hi_r2:+.4f}]")
    print(f"  ΔMAE (fs - pre), Pa    = {m_ma:+.3f}  95% CI [{lo_ma:+.3f}, {hi_ma:+.3f}]")
    print(f"  cars where from-scratch >= pretrained R2: {(fr2 >= pr2).mean()*100:.1f}%")
    print(f"  -> {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
