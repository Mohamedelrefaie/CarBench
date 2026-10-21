#!/usr/bin/env python
"""Aggregate the full-resolution evaluation into a CSV + a manuscript-ready table.

Two result families are combined:
  <model>.json      sweep 1 — sub10k / interp(kNN-IDW) / patch on all 1,154 test cars
  <model>_rbf.json  sweep 2 — interp(RBF), the manuscript's original upscaling,
                    on a 61-car stratified subset

The headline table is reported on the MATCHED subset (all three protocols on the
same geometries, so the comparison is apples-to-apples); full-test-set numbers for
the two cheap protocols are printed alongside to show the subset is representative.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["pointnet", "fno", "pointmae", "pointnet_large", "regdgcnn",
         "point_transformer", "tripnet", "transolver_plus", "transolver",
         "transolver_large", "abupt"]
PRETTY = {"pointnet": "PointNet", "fno": "NeuralOperator", "pointmae": "PointMAE",
          "pointnet_large": "PointNetLarge", "regdgcnn": "RegDGCNN",
          "point_transformer": "PointTransformer", "tripnet": "TripNet",
          "transolver_plus": "Transolver++", "transolver": "Transolver",
          "transolver_large": "TransolverLarge", "abupt": r"AB-UPT\textsuperscript{*}"}
# models whose decoder evaluates arbitrary query points directly (no interpolation)
QUERY_BASED = {"abupt", "tripnet"}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def mean_of(rows, proto, field):
    vals = [r[proto][field] for r in rows if proto in r and r[proto] is not None]
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "full_res"))
    args = ap.parse_args()

    subset_ids = set()
    p = os.path.join(HERE, "rbf_subset.txt")
    if os.path.exists(p):
        subset_ids = {l.strip() for l in open(p) if l.strip()}

    recs = []
    # abupt is evaluated from the author-released checkpoint (AB-UPT*), which is
    # the variant reported in the main benchmark table; its results carry the
    # "_star" tag. See README/response letter for the TripNet provenance note.
    FILE = {"abupt": "abupt_star"}
    for m in ORDER:
        stem = FILE.get(m, m)
        d1 = load(os.path.join(args.results_dir, f"{stem}.json"))
        d2 = load(os.path.join(args.results_dir, f"{stem}_rbf.json"))
        if d1 is None:
            print(f"  (missing sweep1 for {m})")
            continue
        rows = d1["per_car"]
        sub = [r for r in rows if r["car"] in subset_ids] if subset_ids else rows
        rec = {"model": m, "n_full": len(rows), "n_subset": len(sub)}
        # matched subset (headline)
        for proto in ("sub10k", "interp", "patch"):
            for f in ("rel_l2", "r2", "mae", "rmse"):
                rec[f"{proto}_{f}"] = mean_of(sub, proto, f)
        # RBF interpolation on the same subset
        if d2 is not None:
            for f in ("rel_l2", "r2", "mae", "rmse"):
                rec[f"interp_rbf_{f}"] = mean_of(d2["per_car"], "interp", f)
        # full test set, cheap protocols
        for proto in ("sub10k", "patch"):
            for f in ("rel_l2", "r2"):
                rec[f"full_{proto}_{f}"] = mean_of(rows, proto, f)
        rec["query_based"] = m in QUERY_BASED
        recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(f"{args.out_prefix}_table.csv", index=False)
    print(df[["model", "n_full", "n_subset", "sub10k_rel_l2",
              "interp_rbf_rel_l2", "patch_rel_l2"]].to_string(index=False))

    # ---- manuscript table ----
    def fmt(v, nd=4):
        return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"

    lines = []
    for _, r in df.iterrows():
        interp = r.get("interp_rbf_rel_l2", np.nan)
        interp_r2 = r.get("interp_rbf_r2", np.nan)
        note = r"$^{\dagger}$" if r["query_based"] else ""
        lines.append(
            f"{PRETTY.get(r['model'], r['model'])}{note} & "
            f"{fmt(r['sub10k_rel_l2'])} & {fmt(interp)} & {fmt(r['patch_rel_l2'])} & "
            f"{fmt(r['sub10k_r2'])} & {fmt(interp_r2)} & {fmt(r['patch_r2'])} \\\\")
    body = "\n".join(lines)
    tex = r"""\begin{table*}[h!]
\centering
\small
\caption{\textcolor{cyan}{Subsampled versus full-resolution evaluation on a stratified subset of the
held-out test split, with all three protocols applied to the same geometries. \emph{Subsampled}
scores a $10{,}000$-point prediction on those points; \emph{Interpolated} upscales that same
prediction to every node of the full CFD surface mesh ($3.7$--$5.5\times10^{5}$ nodes, mean
$455{,}000$) with the radial-basis interpolation;
\emph{Patchwise} instead predicts every node directly using $\lceil N/10^4 \rceil$ disjoint
$10{,}000$-point passes ($37$--$55$ per geometry), and therefore involves no interpolation.
Patchwise evaluation recovers almost all of the apparent degradation, showing that the loss attributed to resolution is dominated by interpolation error.
$^{\dagger}$AB-UPT\textsuperscript{*} and TripNet decode arbitrary query points directly, so their
patchwise column evaluates every node from a single geometry encoding, with queries chunked only to
fit memory, rather than $\lceil N/10^4 \rceil$ independent passes. They are held to the identical
interpolated protocol as every other model, with the prediction restricted to the same $10{,}000$
points and upscaled in the same way, so that the column compares evaluation protocols rather than
architectures.}}
\label{tab:surface_fullmesh_comparison}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{lccccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Relative L2}} & \multicolumn{3}{c}{$\mathbf{R^2}$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Model} & \textbf{Subsampled} & \textbf{Interpolated} & \textbf{Patchwise}
               & \textbf{Subsampled} & \textbf{Interpolated} & \textbf{Patchwise} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f"{args.out_prefix}_table.tex", "w").write(tex)
    print(f"\nwrote {args.out_prefix}_table.csv and .tex")


if __name__ == "__main__":
    main()
