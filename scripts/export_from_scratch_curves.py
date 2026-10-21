#!/usr/bin/env python
"""Export the AB-UPT and TripNet from-scratch training curves into the layout
that plot_all_loss_curves.py reads (<results>/<model>/learning_curves/training_curves.csv).

The nine other benchmark models are trained by the common harness, which writes
that CSV itself. AB-UPT and TripNet were trained by their own scripts and record
progress differently, so their curves have to be transcribed:

  AB-UPT   results_from_scratch/abupt/train_history.json
           full history - train_loss, val_loss (EMA weights), lr, 30 epochs.
  TripNet  results_from_scratch/tripnet/logs/predictor.log
           lines "epoch N/200 mse X ..." - TRAIN loss only, 200 epochs, and only
           for the final predictor stage (stage-1 joint fit and stage-2 triplane
           fitting are separate). No validation loss was ever logged, so val_loss
           is written empty and the plotting script leaves that curve blank.

    python scripts/export_from_scratch_curves.py --results-dir results_backup
"""
import argparse
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABUPT = os.path.join(HERE, "results_from_scratch", "abupt", "train_history.json")
TRIPNET = os.path.join(HERE, "results_from_scratch", "tripnet", "logs", "predictor.log")
HEADER = ["epoch", "train_loss", "val_loss", "learning_rate"]


def _write(results_dir, model, rows):
    d = os.path.join(results_dir, model, "learning_curves")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "training_curves.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"  {model:8s} -> {path}  ({len(rows)} epochs)")


def export_abupt(results_dir):
    if not os.path.exists(ABUPT):
        print(f"  abupt    skipped (no {ABUPT})")
        return
    h = json.load(open(ABUPT))
    rows = [[i, t, v, lr] for i, (t, v, lr) in
            enumerate(zip(h["train_loss"], h["val_loss"], h["lr"]), start=1)]
    _write(results_dir, "abupt", rows)


def export_tripnet(results_dir):
    if not os.path.exists(TRIPNET):
        print(f"  tripnet  skipped (no {TRIPNET})")
        return
    text = open(TRIPNET).read()
    # val_loss deliberately left empty: the predictor stage logged train MSE only.
    rows = [[int(e), float(m), "", ""] for e, m in
            re.findall(r"epoch\s+(\d+)/\d+\s+mse\s+([0-9.]+)", text)]
    _write(results_dir, "tripnet", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_backup"),
                    help="results tree plot_all_loss_curves.py reads (default: results_backup)")
    args = ap.parse_args()
    print(f"exporting from-scratch curves into {args.results_dir}")
    export_abupt(args.results_dir)
    export_tripnet(args.results_dir)


if __name__ == "__main__":
    main()
