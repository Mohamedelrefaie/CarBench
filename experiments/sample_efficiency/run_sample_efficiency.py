#!/usr/bin/env python3
"""R3.5 sample-efficiency: train Transolver / Transolver++ on increasing fractions of the
train split (10/25/50/100%) and evaluate on the FULL test set, to separate the effect of
dataset size from out-of-distribution difficulty.

One process per GPU (each pinned via CUDA_VISIBLE_DEVICES so the wrapper does not wrap in
DataParallel and so we only use the GPUs that have free memory). Each job trains + evaluates
through the standard scripts/train_and_analyze.py, writing test metrics to
results_sample_efficiency/<model>_frac<F>/comprehensive_benchmark_summary.csv.

  python experiments/sample_efficiency/run_sample_efficiency.py --gpus 1,3 \
      [--models transolver,transolver_plus] [--fracs 10,25,50,100] [--epochs 100 --batch 16]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "PressureVTK", "Dataverse"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--models", default="transolver,transolver_plus")
    ap.add_argument("--fracs", default="10,25,50,100")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-dir", default=os.path.join(REPO, "results_sample_efficiency"))
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    fracs = [int(f) for f in args.fracs.split(",")]
    os.makedirs(args.save_dir, exist_ok=True)

    # job queue: smallest fractions first (fast feedback)
    jobs = [(m, f) for f in fracs for m in models]
    jobs.sort(key=lambda x: x[1])
    print(f"{len(jobs)} jobs on GPUs {gpus}: {jobs}")

    running = {}   # gpu -> (proc, job, start, logf)
    results = {}

    def launch(job, gpu):
        model, frac = job
        save = os.path.join(args.save_dir, f"{model}_frac{frac}")
        os.makedirs(save, exist_ok=True)
        log = os.path.join(save, "train.log")
        cmd = ["python", "scripts/train_and_analyze.py", model,
               "--dataset-path", DS,
               "--subset-dir", os.path.join(HERE, "subsets", f"frac{frac}"),
               "--config", os.path.join(REPO, "models", model, "config.yaml"),
               "--save-dir", save, "--epochs", str(args.epochs), "--batch-size", str(args.batch)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        lf = open(log, "w")
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(f"  [GPU {gpu}] {model} frac{frac} (pid {p.pid}) -> {log}", flush=True)
        running[gpu] = (p, job, time.time(), lf)

    while jobs or running:
        for gpu in gpus:
            if gpu not in running and jobs:
                launch(jobs.pop(0), gpu)
        time.sleep(10)
        for gpu, (p, job, t0, lf) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                lf.close()
                results[job] = rc
                dur = (time.time() - t0) / 60
                print(f"  [GPU {gpu}] {job} {'OK' if rc == 0 else 'FAIL rc=' + str(rc)} ({dur:.1f} min)", flush=True)
                del running[gpu]
    print("\n=== summary ===")
    for j, rc in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  {'OK' if rc == 0 else 'FAIL'}  {j[0]} frac{j[1]}")


if __name__ == "__main__":
    main()
