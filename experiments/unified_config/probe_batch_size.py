#!/usr/bin/env python3
"""Find the largest single batch size that fits EVERY model on the GPU.

For each model: build it once, pull one real training batch at the max candidate
size, then descend candidate batch sizes by slicing the batch and running a real
fwd+bwd+optimizer step (dummy loss — memory is what matters). The first size that
runs without CUDA OOM is that model's max-feasible batch. The unified batch is the
MIN across all models.

Each model runs in its own subprocess so a CUDA OOM in one can't corrupt another's
context, and so each imports its own `model` module cleanly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/unified_config/probe_batch_size.py
    # internal: ... probe_batch_size.py --model regdgcnn   (prints 'MAXBATCH <n>')
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

MODELS = ["transolver", "transolver_large", "transolver_plus", "regdgcnn",
          "point_transformer", "pointnet", "pointnet_large", "pointmae", "fno"]
CANDIDATES = [32, 16, 8, 4, 2, 1]


def _to_dev(x, dev):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_dev(v, dev) for v in x)
    return x


def _slice0(x, b):
    import torch
    if isinstance(x, torch.Tensor):
        return x[:b]
    if isinstance(x, (list, tuple)):
        return type(x)(_slice0(v, b) for v in x)
    return x


def probe_model(name: str) -> int:
    """Return the max feasible batch size for one model (0 if none)."""
    import torch
    logging.disable(logging.INFO)
    from scripts.train_and_analyze import load_model_wrapper

    cfg = yaml.safe_load(open(os.path.join(HERE, "configs", f"{name}.yaml")))
    cfg["dataset_path"] = os.environ["DATASET_PATH"]
    cfg["subset_dir"] = os.environ["SUBSET_DIR"]
    cfg["batch_size"] = CANDIDATES[0]
    cfg["num_workers"] = 0

    wrapper = load_model_wrapper(name, cfg, logging.getLogger("x"))
    model = wrapper.load_model()
    dev = wrapper.device
    train_loader = wrapper.get_data_loaders()[0]
    full = next(iter(train_loader))               # one batch at the max candidate
    data_full, _ = wrapper.preprocess_batch(full)

    model.train()
    best = 0
    for b in CANDIDATES:
        try:
            data = _to_dev(_slice0(data_full, b), dev)
            opt = wrapper.get_optimizer(model)
            opt.zero_grad(set_to_none=True)
            out = wrapper.forward(model, data)
            out = out[0] if isinstance(out, (list, tuple)) else out
            loss = out.float().pow(2).mean()       # dummy loss → real backward graph
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            best = b
            del out, loss, data, opt
            torch.cuda.empty_cache()
            break                                  # descending → first success is the max
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()

    if args.model:
        print(f"MAXBATCH {probe_model(args.model)}")
        return

    assert os.environ.get("DATASET_PATH") and os.environ.get("SUBSET_DIR"), \
        "set DATASET_PATH and SUBSET_DIR"

    results = {}
    print(f"{'model':<18}{'max feasible batch':>20}")
    print("-" * 40)
    for name in MODELS:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--model", name],
            cwd=REPO_ROOT, env=dict(os.environ), capture_output=True, text=True,
        )
        val = 0
        for line in p.stdout.splitlines():
            if line.startswith("MAXBATCH"):
                val = int(line.split()[1])
        if val == 0:
            tail = (p.stderr.strip().splitlines() or ["?"])[-1]
            print(f"{name:<18}{'ERROR':>20}  {tail[:60]}")
        else:
            print(f"{name:<18}{val:>20}")
        results[name] = val

    feasible = [v for v in results.values() if v > 0]
    unified = min(feasible) if feasible else 1
    print("-" * 40)
    print(f"UNIFIED BATCH = {unified}  (= min across models)")

    # Record the result; generate_unified_configs.py reads this to set batch_size
    # (keeps the documented unified_training.yaml recipe intact).
    import json
    json.dump(results | {"unified_batch": unified},
              open(os.path.join(HERE, "batch_probe_results.json"), "w"), indent=2)
    print(f"wrote batch_probe_results.json (unified_batch={unified}); "
          f"now re-run generate_unified_configs.py to bake it into the configs")


if __name__ == "__main__":
    main()
