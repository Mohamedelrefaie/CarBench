#!/usr/bin/env python3
"""R3.5 cross-category (OOD) sample efficiency: train Transolver / Transolver++ on the
N+F category pool subsampled to increasing size, test on the held-out E (Estateback) category.

This separates dataset SIZE from OOD difficulty: the training categories are FIXED (N+F, never E),
so any change in E (OOD) performance with training size is the pure size effect. The full-pool
point (5388 N+F cars -> E) is the paper's existing cross-category result; we add the smaller-size
points on the IDENTICAL recipe (native config, batch 8, 200 epochs early-stopped) and test set
(all 1385 E cars, results/cross_category_generalization/category_splits/train_NF_test_E).

  python experiments/sample_efficiency/run_ood_sample_efficiency.py --gpus 1,2,3
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
    ap.add_argument("--gpus", default="1,2,3")
    ap.add_argument("--models", default="transolver,transolver_plus")
    ap.add_argument("--sizes", default="600,1500,3000")
    ap.add_argument("--epochs", type=int, default=200)   # matches the cross-category runs
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-dir", default=os.path.join(REPO, "results_sample_efficiency", "ood"))
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.sizes.split(",")]
    os.makedirs(args.save_dir, exist_ok=True)

    jobs = [(m, n) for n in sizes for m in models]
    jobs.sort(key=lambda x: x[1])
    print(f"{len(jobs)} OOD jobs (N+F -> test E) on GPUs {gpus}: {jobs}", flush=True)
    running, results = {}, {}

    def launch(job, gpu):
        model, n = job
        save = os.path.join(args.save_dir, f"{model}_nf{n}")
        os.makedirs(save, exist_ok=True)
        cmd = ["python", "scripts/train_and_analyze.py", model,
               "--dataset-path", DS,
               "--subset-dir", os.path.join(HERE, "ood_subsets", f"nf{n}"),
               "--config", os.path.join(REPO, "models", model, "config.yaml"),
               "--save-dir", save, "--epochs", str(args.epochs), "--batch-size", str(args.batch)]
        lf = open(os.path.join(save, "train.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(f"  [GPU {gpu}] {model} nf{n} (pid {p.pid})", flush=True)
        running[gpu] = (p, job, time.time(), lf)

    while jobs or running:
        for gpu in gpus:
            if gpu not in running and jobs:
                launch(jobs.pop(0), gpu)
        time.sleep(10)
        for gpu, (p, job, t0, lf) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                lf.close(); results[job] = rc
                print(f"  [GPU {gpu}] {job} {'OK' if rc == 0 else 'rc=' + str(rc)} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                del running[gpu]
    print("\n=== summary ===")
    for j, rc in sorted(results.items()):
        print(f"  {'OK' if rc == 0 else 'FAIL'}  {j[0]} nf{j[1]}")


if __name__ == "__main__":
    main()
