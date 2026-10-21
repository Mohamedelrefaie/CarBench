#!/usr/bin/env python3
"""Aggregate the R3.5 sample-efficiency sweep into a curve/table.

Reads results_sample_efficiency/<model>_frac<F>/comprehensive_benchmark_summary.csv
(test_r2/test_mae/test_rmse/test_rel_l2 on the FULL test set) and writes, per model, the
test metric vs. training-set fraction. Output: sample_efficiency_table.csv + .tex + a console
summary. Test set is held fixed (full 1,153 cars); only the training-set size varies.
"""
from __future__ import annotations

import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "results_sample_efficiency")
FRAC_CARS = {10: 582, 25: 1455, 50: 2910, 100: 5819}
DISPLAY = {"transolver": "Transolver", "transolver_plus": "Transolver++"}
MODELS = ["transolver", "transolver_plus"]
FRACS = [10, 25, 50, 100]


def read_row(model, frac):
    d = os.path.join(RESULTS, f"{model}_frac{frac}")
    csvs = glob.glob(os.path.join(d, "comprehensive_benchmark_summary.csv")) \
        or glob.glob(os.path.join(d, "**", "comprehensive_benchmark_summary.csv"), recursive=True)
    if not csvs:
        return None
    rows = list(csv.DictReader(open(csvs[0])))
    if not rows:
        return None
    r = rows[-1]
    def g(k):
        try:
            return float(r.get(k, ""))
        except (TypeError, ValueError):
            return None
    return {"r2": g("test_r2"), "mae": g("test_mae"), "rmse": g("test_rmse"),
            "rel_l2": g("test_rel_l2"), "epochs": r.get("epochs_trained", "")}


def main():
    data = {m: {f: read_row(m, f) for f in FRACS} for m in MODELS}

    print("R3.5 sample efficiency — test metrics vs. training-set size (full test set, m^2/s^2)\n")
    for m in MODELS:
        print(f"=== {DISPLAY[m]} ===")
        print(f"{'train %':>8}{'#cars':>8}{'R2':>9}{'MAE':>9}{'RMSE':>9}{'RelL2':>9}")
        for f in FRACS:
            r = data[m][f]
            if r is None:
                print(f"{f:>7}%{FRAC_CARS[f]:>8}{'pending':>36}")
            else:
                fr = lambda x, p=4: f"{x:.{p}f}" if isinstance(x, float) else "--"
                print(f"{f:>7}%{FRAC_CARS[f]:>8}{fr(r['r2']):>9}{fr(r['mae'],2):>9}"
                      f"{fr(r['rmse'],2):>9}{fr(r['rel_l2']):>9}")
        print()

    # CSV
    out_csv = os.path.join(HERE, "sample_efficiency_table.csv")
    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["model", "train_frac_pct", "train_cars", "test_R2", "test_MAE",
                    "test_RMSE", "test_RelL2"])
        for m in MODELS:
            for f in FRACS:
                r = data[m][f] or {}
                w.writerow([DISPLAY[m], f, FRAC_CARS[f], r.get("r2", ""), r.get("mae", ""),
                            r.get("rmse", ""), r.get("rel_l2", "")])

    # LaTeX (R2 vs fraction, both models)
    out_tex = os.path.join(HERE, "sample_efficiency_table.tex")
    def tf(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "--"
    with open(out_tex, "w") as ftex:
        ftex.write("% R3.5 sample efficiency: test R^2 (and Rel-L2) vs training-set size, full test set.\n")
        ftex.write("\\begin{tabular}{lr" + "rr" * len(FRACS) + "}\n\\toprule\n")
        ftex.write("Model & metric & " + " & ".join(f"{f}\\%" for f in FRACS) + " \\\\\n")
        ftex.write("& \\#train cars & " + " & ".join(str(FRAC_CARS[f]) for f in FRACS) + " \\\\\n\\midrule\n")
        for m in MODELS:
            r2s = " & ".join(tf((data[m][f] or {}).get("r2")) for f in FRACS)
            rl2 = " & ".join(tf((data[m][f] or {}).get("rel_l2")) for f in FRACS)
            ftex.write(f"{DISPLAY[m]} & $R^2$ & {r2s} \\\\\n")
            ftex.write(f" & Rel-$L_2$ & {rl2} \\\\\n")
        ftex.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {os.path.relpath(out_csv, REPO)}, {os.path.relpath(out_tex, REPO)}")


if __name__ == "__main__":
    main()
