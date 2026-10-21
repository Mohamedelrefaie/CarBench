#!/usr/bin/env python3
"""Aggregate the R3.5 cross-category (OOD) sample-efficiency curve.

OOD setting: train on the N+F category pool (never E), test on held-out E (Estateback).
Curve = test metric on E vs. N+F training-set size. The full-pool point (5388) is the paper's
existing cross-category result (results/cross_category_generalization/<model>/train_NF_test_E).
The smaller-size points come from results_sample_efficiency/ood/<model>_nf<N>.

If the OOD metric improves with training size, dataset SIZE matters; the residual gap to the
in-distribution-E reference (full model evaluated on E test cars) is the pure OOD penalty,
separated from size.
"""
from __future__ import annotations

import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OOD = os.path.join(REPO, "results_sample_efficiency", "ood")
CC = os.path.join(REPO, "results", "cross_category_generalization")
MODELS = ["transolver", "transolver_plus"]
DISP = {"transolver": "Transolver", "transolver_plus": "Transolver++"}
SIZES = [600, 1500, 3000]
FULL_NF = 5388
KEYS = ("test_r2", "test_mae", "test_rmse", "test_rel_l2", "test_rel_l1",
        "test_p50_error", "test_p95_error", "test_p99_error")


def read_summary(path):
    csvs = glob.glob(path) or glob.glob(os.path.join(os.path.dirname(path), "**",
                                                       "comprehensive_benchmark_summary.csv"), recursive=True)
    if not csvs:
        return None
    rows = list(csv.DictReader(open(csvs[0])))
    if not rows:
        return None
    r = rows[-1]
    def g(k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return None
    return {k: g(k) for k in KEYS}


def main():
    data = {m: {} for m in MODELS}
    for m in MODELS:
        for n in SIZES:
            data[m][n] = read_summary(os.path.join(OOD, f"{m}_nf{n}", "comprehensive_benchmark_summary.csv"))
        data[m][FULL_NF] = read_summary(os.path.join(CC, m, "train_NF_test_E", "comprehensive_benchmark_summary.csv"))

    xs = SIZES + [FULL_NF]
    print("R3.5 cross-category (OOD) sample efficiency: train N+F (subsampled) -> test E (held out)\n")
    for m in MODELS:
        print(f"=== {DISP[m]} (OOD: test on E) ===")
        print(f"{'N+F train':>11}{'R2':>9}{'MAE':>9}{'RMSE':>9}{'RelL2':>9}{'P95':>9}")
        for n in xs:
            r = data[m][n]
            tag = " (paper)" if n == FULL_NF else ""
            if r is None:
                print(f"{n:>11}{'pending':>9}{tag}")
            else:
                fm = lambda x, p=4: f"{x:.{p}f}" if isinstance(x, float) else "--"
                print(f"{n:>11}{fm(r['test_r2']):>9}{fm(r['test_mae'],2):>9}{fm(r['test_rmse'],2):>9}"
                      f"{fm(r['test_rel_l2']):>9}{fm(r['test_p95_error'],1):>9}{tag}")
        print()

    out_csv = os.path.join(HERE, "ood_sample_efficiency_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "nf_train_cars", "is_paper_endpoint"] + list(KEYS))
        for m in MODELS:
            for n in xs:
                r = data[m][n] or {}
                w.writerow([DISP[m], n, n == FULL_NF] + [r.get(k, "") for k in KEYS])

    out_tex = os.path.join(HERE, "ood_sample_efficiency_table.tex")
    def tf(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "--"
    with open(out_tex, "w") as f:
        f.write("% R3.5 cross-category OOD sample efficiency: test R^2 on held-out E vs N+F train size.\n")
        f.write("\\begin{tabular}{lr" + "rr" * len(xs) + "}\n\\toprule\n")
        f.write("Model & metric & " + " & ".join(str(n) for n in xs) + " \\\\\n")
        f.write("& N+F train cars & & & & \\\\\n\\midrule\n")
        for m in MODELS:
            r2 = " & ".join(tf((data[m][n] or {}).get("test_r2")) for n in xs)
            rl2 = " & ".join(tf((data[m][n] or {}).get("test_rel_l2")) for n in xs)
            f.write(f"{DISP[m]} & $R^2$ (E, OOD) & {r2} \\\\\n")
            f.write(f" & Rel-$L_2$ & {rl2} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {os.path.relpath(out_csv, REPO)}, {os.path.relpath(out_tex, REPO)}")


if __name__ == "__main__":
    main()
