#!/usr/bin/env python3
"""Train all 9 models under the unified config — 4-GPU job queue.

Keeps one training per GPU; launches the next model when a GPU frees. Each model is
trained+evaluated by the proven entry point (scripts/train_and_analyze.py) with its
unified config, writing to results_unified/<name>/.

Usage:
    CUDA_VISIBLE_DEVICES unused — pass GPUs explicitly:
    python experiments/unified_config/run_unified_training.py \
        --dataset-path $DATASET_PATH --subset-dir $SUBSET_DIR \
        --gpus 0,1,2,3 [--epochs N] [--models a,b] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_MODELS = ["transolver", "transolver_large", "transolver_plus", "regdgcnn",
                  "point_transformer", "pointnet", "pointnet_large", "pointmae", "fno"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--subset-dir", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--save-dir", default=os.path.join(REPO_ROOT, "results_unified"))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (e.g. 1 for smoke)")
    ap.add_argument("--smoke", action="store_true", help="1-epoch smoke run")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    queue = [m.strip() for m in args.models.split(",") if m.strip()]
    epochs = 1 if args.smoke else args.epochs
    os.makedirs(args.save_dir, exist_ok=True)

    running = {}   # gpu -> (proc, model, start, logfile)
    results = {}
    print(f"Training {len(queue)} models on GPUs {gpus} -> {args.save_dir}")

    def launch(model, gpu):
        cfg = os.path.join(HERE, "configs", f"{model}.yaml")
        save = os.path.join(args.save_dir, model)
        os.makedirs(save, exist_ok=True)
        log = os.path.join(save, "train.log")
        cmd = ["python", "scripts/train_and_analyze.py", model,
               "--dataset-path", args.dataset_path, "--subset-dir", args.subset_dir,
               "--config", cfg, "--save-dir", save]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        lf = open(log, "w")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(f"  [GPU {gpu}] launched {model} (pid {proc.pid}) -> {log}")
        running[gpu] = (proc, model, time.time(), lf)

    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                launch(queue.pop(0), gpu)
        time.sleep(5)
        for gpu, (proc, model, start, lf) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                lf.close()
                dur = (time.time() - start) / 60
                ok = rc == 0
                results[model] = ok
                print(f"  [GPU {gpu}] {model} {'✅' if ok else '❌ rc='+str(rc)} ({dur:.1f} min)")
                del running[gpu]

    print("\n=== summary ===")
    for m in (m.strip() for m in args.models.split(",")):
        print(f"  {'✅' if results.get(m) else '❌'} {m}")
    n_ok = sum(1 for v in results.values() if v)
    print(f"{n_ok}/{len(results)} succeeded -> {args.save_dir}")


if __name__ == "__main__":
    main()
