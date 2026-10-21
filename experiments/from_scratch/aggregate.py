#!/usr/bin/env python3
"""From-scratch vs pre-trained comparison (rebuttal point 2), in CarBench metric convention.

Reports the CarBench (arXiv:2512.07847, the DrivAerNet++ benchmark) primary metric set --
MSE, MAE, RMSE, R^2, Relative L2 -- in **denormalized kinematic pressure units (m^2/s^2)**
(NOT Pascals; CarBench reports raw OpenFOAM kinematic pressure). The pre-trained reference
values are CarBench Table 1; the from-scratch values are our runs evaluated identically.

Inputs:
  - AB-UPT from-scratch: results_from_scratch/abupt/eval_from_scratch.json (fp32 eval_abupt.py)
  - AB-UPT pretrained (fp32 re-eval, sanity): results_from_scratch/abupt/eval_pretrained.json
  - TripNet from-scratch: results_from_scratch/tripnet/tripnet_metrics.json (denormalized r2/mae/rmse/relL2)

Writes from_scratch_comparison.csv, from_scratch_comparison.tex.
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# CarBench Table 1 (test set, m^2/s^2, +/-95% bootstrap CI dropped here). These ARE the
# "pre-trained / provided" rows our from-scratch runs are compared against.
CARBENCH = {
    "AB-UPT":  {"mse": 559, "mae": 10.81, "rmse": 23.6, "r2": 0.9675, "rel_l2": 0.1358},
    "TripNet": {"mse": 596, "mae": 13.30, "rmse": 24.9, "r2": 0.9590, "rel_l2": 0.1608},
}


def _mse(rmse):
    return round(rmse * rmse, 1) if isinstance(rmse, (int, float)) else None


_KEYS = ("mse", "mae", "rmse", "r2", "max_error", "rel_l1", "rel_l2",
         "p50_error", "p90_error", "p95_error", "p99_error")


def _pick(d):
    return {k: d.get(k) for k in _KEYS}


def read_abupt_eval(tag):
    p = os.path.join(REPO, "results_from_scratch", "abupt", f"eval_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = _pick(d)
    out.update(n_cars=d.get("n_cars"), nonfinite=d.get("nonfinite_cars"))
    return out


def read_tripnet_from_scratch():
    p = os.path.join(REPO, "results_from_scratch", "tripnet", "tripnet_metrics.json")
    if not os.path.exists(p):
        return None
    return _pick(json.load(open(p)))   # denormalized (m^2/s^2) metrics, matches CarBench


def f(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "pending"


COLS = [("MSE", "mse", 0), ("MAE", "mae", 2), ("RMSE", "rmse", 2),
        ("R2", "r2", 4), ("RelL1", "rel_l1", 4), ("RelL2", "rel_l2", 4)]
PCTS = [("P50", "p50_error"), ("P90", "p90_error"), ("P95", "p95_error"),
        ("P99", "p99_error"), ("max", "max_error")]
ALL_KEYS = ["mse", "mae", "rmse", "r2", "max_error", "rel_l1", "rel_l2",
            "p50_error", "p90_error", "p95_error", "p99_error"]


def row_str(d):
    return "".join(f"{f(d.get(k), p):>10}" for _, k, p in COLS) if d else f"{'pending':>60}"


def pct_str(d):
    if not d:
        return ""
    return "  ".join(f"{nm}={f(d.get(k), 1)}" for nm, k in PCTS)


def main():
    ab = read_abupt_eval("from_scratch")
    ab_pre = read_abupt_eval("pretrained")
    tn = read_tripnet_from_scratch()

    print("From-scratch vs pre-trained (CarBench metric set, kinematic pressure m^2/s^2)")
    print("CarBench Table 1 reports MSE/MAE/RMSE/R2/RelL2; RelL1 + percentiles are our full set.\n")
    hdr = "".join(f"{name:>10}" for name, _, _ in COLS)
    for model, fs in (("AB-UPT", ab), ("TripNet", tn)):
        print(f"=== {model} ===")
        print(f"{'variant':<26}{hdr}")
        print(f"{'pre-trained (CarBench)':<26}{row_str(CARBENCH[model])}")
        if model == "AB-UPT" and ab_pre:
            print(f"{'pre-trained (our fp32)':<26}{row_str(ab_pre)}    |err| {pct_str(ab_pre)}")
        print(f"{'from scratch':<26}{row_str(fs)}    |err| {pct_str(fs)}")
        print()

    # CSV (full metric set)
    out_csv = os.path.join(HERE, "from_scratch_comparison.csv")
    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["Model", "Init"] + ALL_KEYS + ["units"])
        for model, fs in (("AB-UPT", ab), ("TripNet", tn)):
            w.writerow([model, "pretrained_carbench"] + [CARBENCH[model].get(k, "") for k in ALL_KEYS] + ["m2/s2"])
            if model == "AB-UPT" and ab_pre:
                w.writerow([model, "pretrained_our_fp32"] + [ab_pre.get(k, "") for k in ALL_KEYS] + ["m2/s2"])
            if fs:
                w.writerow([model, "from_scratch"] + [fs.get(k, "") for k in ALL_KEYS] + ["m2/s2"])

    # LaTeX
    out_tex = os.path.join(HERE, "from_scratch_comparison.tex")
    def tf(x, p=2):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "--"
    with open(out_tex, "w") as ftex:
        ftex.write("% From-scratch vs pre-trained (CarBench metrics, kinematic pressure m^2/s^2).\n")
        ftex.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        ftex.write("Model & Init & MSE & MAE & RMSE & $R^2$ & Rel-$L_2$ \\\\\n\\midrule\n")
        for model, fs in (("AB-UPT", ab), ("TripNet", tn)):
            cb = CARBENCH[model]
            ftex.write(f"{model} & pre-trained & {tf(cb['mse'],0)} & {tf(cb['mae'])} & "
                       f"{tf(cb['rmse'])} & {tf(cb['r2'],4)} & {tf(cb['rel_l2'],4)} \\\\\n")
            if fs:
                ftex.write(f"{model} & from scratch & {tf(fs['mse'],0)} & {tf(fs['mae'])} & "
                           f"{tf(fs['rmse'])} & {tf(fs['r2'],4)} & {tf(fs['rel_l2'],4)} \\\\\n")
        ftex.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {os.path.relpath(out_csv, REPO)}, {os.path.relpath(out_tex, REPO)}")


if __name__ == "__main__":
    main()
