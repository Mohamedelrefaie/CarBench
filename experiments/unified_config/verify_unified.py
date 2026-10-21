#!/usr/bin/env python3
"""Verify every model honors the unified recipe BEFORE the long run.

For each model: load its unified config, instantiate the wrapper, build the model,
and inspect the optimizer / scheduler / loss / lr it would actually train with.
Asserts Adam + CosineAnnealingLR(T_max=100) + MSELoss + lr=1e-3 for all. Any model
that diverges (e.g., a wrapper without a 'cosine' branch) is reported so we can pick
a scheduler all wrappers support.

Usage:
    python experiments/unified_config/verify_unified.py
"""
from __future__ import annotations

import logging
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

from scripts.train_and_analyze import load_model_wrapper  # noqa: E402

logging.disable(logging.INFO)
MODELS = ["transolver", "transolver_large", "transolver_plus", "regdgcnn",
          "point_transformer", "pointnet", "pointnet_large", "pointmae", "fno"]


def check(name):
    cfg = yaml.safe_load(open(os.path.join(HERE, "configs", f"{name}.yaml")))
    cfg["dataset_path"] = "/tmp/dummy"
    cfg["subset_dir"] = "/tmp/dummy"
    wrapper = load_model_wrapper(name, cfg, logging.getLogger("x"))
    model = wrapper.load_model()
    opt = wrapper.get_optimizer(model)
    sched = wrapper.get_scheduler(opt)
    loss = wrapper.get_loss_function()
    lr = opt.param_groups[0]["lr"]
    wd = opt.param_groups[0].get("weight_decay", None)
    return {
        "optimizer": type(opt).__name__,
        "scheduler": type(sched).__name__ if sched is not None else "None",
        "T_max": getattr(sched, "T_max", None),
        "loss": type(loss).__name__ if hasattr(loss, "__class__") else str(loss),
        "lr": lr,
        "weight_decay": wd,
    }


def _check_one_subprocess(name):
    """Run check(name) in a clean subprocess (each model imports its own `model`
    module; running all in one process collides on the cached generic name)."""
    import json
    import subprocess
    code = (
        "import json,logging;logging.disable(logging.INFO);"
        "import sys;sys.path.insert(0,%r);"
        "from experiments.unified_config.verify_unified import check;"
        "print('RESULT'+json.dumps(check(%r)))" % (REPO_ROOT, name)
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise RuntimeError((p.stderr.strip().splitlines() or ["unknown error"])[-1])


def main():
    print(f"{'model':<18}{'optimizer':<10}{'scheduler':<24}{'loss':<12}{'lr':<8}{'wd':<6}")
    print("-" * 82)
    ok = True
    for name in MODELS:
        try:
            r = _check_one_subprocess(name)
        except Exception as e:
            print(f"{name:<18}ERROR: {e}")
            ok = False
            continue
        good = (r["optimizer"] == "Adam"
                and r["scheduler"] == "CosineAnnealingLR"
                and "MSELoss" in r["loss"]
                and abs(r["lr"] - 1e-3) < 1e-9)
        flag = "" if good else "  <-- DIVERGES"
        if not good:
            ok = False
        sched = f"{r['scheduler']}(T={r['T_max']})"
        print(f"{name:<18}{r['optimizer']:<10}{sched:<24}{r['loss']:<12}"
              f"{r['lr']:<8}{str(r['weight_decay']):<6}{flag}")
    print("-" * 82)
    print("ALL UNIFIED ✅" if ok else "SOME MODELS DIVERGE — adjust recipe (see above) ❌")


if __name__ == "__main__":
    main()
