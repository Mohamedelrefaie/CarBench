#!/usr/bin/env python
"""Train AB-UPT (SurfaceOnlyABUPT) FROM SCRATCH on DrivAerNet, mirroring the Noether
recipe (Emmi-AI/noether recipes/aero_cfd) — rebuttal point 2 (no pretrained weights).

Why a standalone trainer (not scripts/train_and_analyze.py):
  models/abupt/wrapper.py is EVAL-ONLY — its forward() forces model.eval()+torch.no_grad()
  and its denormalize-in-forward + DenormalizedMSELoss double-normalize under UnifiedTrainer.
  So we train here with the model + collator + dataset directly, then EVALUATE the saved
  checkpoint through the UNCHANGED eval path (scripts/train_and_analyze.py abupt
  --evaluate-only --checkpoint ...) so the from-scratch number is computed identically to
  the pretrained AB-UPT row in the paper.

Architecture is matched to the released checkpoint's stored ctor_kwargs
(dim=192, geometry_depth=1, blocks='p', num_surface_blocks=11, num_heads=3,
output_dim_surface=4 -> the vendored SurfaceOnlyABUPT() default, 6.0M params) so the only
difference vs the pretrained row is initialization (random vs public pretrained weights).
Budget (30 epochs, batch 1), EMA 0.9999 and the surface-pressure+friction MSE are all
taken from that checkpoint's metadata / the Noether preset. See ABUPT_RECIPE.md.

Usage:
    python experiments/from_scratch/train_abupt.py \
        --dataset-path $DATASET_PATH --config experiments/from_scratch/configs/abupt.yaml \
        --save-dir results_from_scratch/abupt --gpu 0 [--smoke] [--max-train-batches N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ABUPT_DIR = os.path.join(REPO_ROOT, "models", "abupt")
sys.path.insert(0, ABUPT_DIR)

from surface_only_model import SurfaceOnlyABUPT          # noqa: E402
from surface_abupt_collator import SurfaceOnlyAbuptCollator  # noqa: E402
from drivaernet_dataset import DrivAerNetDataset         # noqa: E402

TARGET_KEYS = {
    "surface_anchor_pressure", "surface_anchor_wallshearstress",
    "surface_query_pressure", "surface_query_wallshearstress",
}


# ----------------------------- Lion optimizer ------------------------------
class Lion(torch.optim.Optimizer):
    """Lion (EvoLved Sign Momentum), Chen et al. 2023 — the AB-UPT/Noether default.

    Decoupled weight decay; update = sign(b1*m + (1-b1)*g); m <- b2*m + (1-b2)*g.
    Sign-based updates are invariant to a constant grad scale, but we still unscale
    before stepping (GradScaler.unscale_) so clipping at 0.25 is in true grad units.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                p.mul_(1 - lr * wd)                                   # decoupled WD
                update = exp_avg.mul(b1).add_(grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(b2).add_(grad, alpha=1 - b2)             # momentum
        return loss


# ------------------------------- EMA ---------------------------------------
class EMA:
    """Exponential moving average of model params (decay 0.9999, as released ckpt)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)  # ints/buffers: track latest

    def state_dict(self, ref_model):
        """EMA weights cast back to the reference model's dtypes."""
        ref = ref_model.state_dict()
        return {k: self.shadow[k].to(ref[k].dtype) for k in ref}


# ------------------------------- Data --------------------------------------
class CarDataset(Dataset):
    """One car -> a collated AB-UPT batch dict. File read + point sampling happen in
    worker processes (the expensive part), so num_workers parallelizes the VTP I/O."""

    def __init__(self, ds, collator):
        self.ds, self.col = ds, collator

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = {
            "surface_position_vtp": self.ds.getitem_surface_position_vtp(idx),
            "surface_pressure": self.ds.getitem_surface_pressure(idx),
            "surface_wallshearstress": self.ds.getitem_surface_wallshearstress(idx),
        }
        return self.col([s])


def _identity_collate(batch):
    return batch[0]  # batch_size=1


def make_loader(root, split, cfg, shuffle, workers):
    ds = DrivAerNetDataset(root=root, split=split, skip_missing=True)
    col = SurfaceOnlyAbuptCollator(
        num_geometry_points=cfg["num_geometry_points"],
        num_geometry_supernodes=cfg["num_geometry_supernodes"],
        num_surface_anchor_points=cfg["num_surface_anchor_points"],
        use_query_positions=False,   # num_surface_queries=0 (Noether preset)
        dataset=ds,
    )
    loader = DataLoader(
        CarDataset(ds, col), batch_size=1, shuffle=shuffle, num_workers=workers,
        collate_fn=_identity_collate, pin_memory=False, persistent_workers=workers > 0,
        drop_last=False,
    )
    return ds, loader


# ----------------------------- train / eval --------------------------------
def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def surface_loss(out, batch):
    """Equal-weight MSE on the normalized surface fields (pressure[1] + friction[3]),
    matching Noether FIELD_WEIGHTS surface_pressure=1.0 + surface_friction=1.0."""
    p = F.mse_loss(out["surface_anchor_pressure"].float(), batch["surface_anchor_pressure"].float())
    w = F.mse_loss(out["surface_anchor_wallshearstress"].float(), batch["surface_anchor_wallshearstress"].float())
    return p + w, p, w


def run_epoch(model, loader, dev, scaler, optimizer, scheduler, ema, clip, train, max_batches=None):
    model.train(train)
    tot, totp, n = 0.0, 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_dev(batch, dev)
        inp = {k: v for k, v in batch.items() if k not in TARGET_KEYS}
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**inp)
                loss, lp, _ = surface_loss(out, batch)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
        else:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**inp)
                loss, lp, _ = surface_loss(out, batch)
        tot += float(loss); totp += float(lp); n += 1
        if train and (i % 200 == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(f"    step {i:5d}/{len(loader)} loss {float(loss):.4f} (p {float(lp):.4f}) "
                  f"lr {lr:.2e} {(time.time()-t0)/max(i,1):.3f}s/it", flush=True)
    return tot / max(n, 1), totp / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--subset-dir", default=None)  # accepted for parity; splits resolved internally
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "abupt.yaml"))
    ap.add_argument("--save-dir", default=os.path.join(REPO_ROOT, "results_from_scratch", "abupt"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, few batches")
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--train-split", default="train", choices=["train", "val", "test", "all"],
                    help="dataset split to train on; 'all' = union of train/val/test (full dataset)")
    ap.add_argument("--no-val", action="store_true",
                    help="skip per-epoch validation and best-checkpoint selection; "
                         "only abupt_from_scratch_last.th is written (use with --train-split all)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = yaml.safe_load(open(args.config))
    epochs = args.epochs or cfg.get("epochs", 30)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(args.save_dir, exist_ok=True)
    dev = "cuda"

    ctor = dict(cfg.get("ctor_kwargs", {})) or dict(
        dim=192, geometry_depth=1, blocks="p", num_surface_blocks=11,
        num_heads=3, radius=1, ndim=3, input_dim=3, output_dim_surface=4)

    print(f"[abupt-from-scratch] gpu={args.gpu} epochs={epochs} seed={seed}")
    print(f"[abupt-from-scratch] ctor_kwargs={ctor}")

    train_ds, train_loader = make_loader(args.dataset_path, args.train_split, cfg, shuffle=True, workers=args.workers)
    if args.no_val:
        val_ds = val_loader = None
        print(f"[abupt-from-scratch] train={len(train_ds)} (split={args.train_split}, no validation)")
    else:
        val_ds, val_loader = make_loader(args.dataset_path, "val", cfg, shuffle=False, workers=max(2, args.workers // 2))
        print(f"[abupt-from-scratch] train={len(train_ds)} val={len(val_ds)}")

    model = SurfaceOnlyABUPT(**{k: v for k, v in ctor.items() if k != "kind"}).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[abupt-from-scratch] params={nparam:,} (random init)")

    optimizer = Lion(model.parameters(), lr=float(cfg["lr"]), betas=tuple(cfg.get("betas", (0.9, 0.99))),
                     weight_decay=float(cfg["weight_decay"]))
    clip = float(cfg.get("gradient_clip", 0.25))

    mtb = 4 if args.smoke else args.max_train_batches
    mvb = 4 if args.smoke else args.max_val_batches
    steps_per_epoch = (mtb if mtb is not None else len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(cfg.get("warmup_percent", 0.05) * total_steps))
    eta_min = float(cfg.get("eta_min", 1e-6))
    base_lr = float(cfg["lr"])

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return (eta_min + (base_lr - eta_min) * cos) / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9999)))

    history = {"train_loss": [], "val_loss": [], "val_loss_ema": [], "lr": []}
    best_val = float("inf"); best_epoch = -1
    t_start = time.time()

    def save_ckpt(name, ema_state):
        path = os.path.join(args.save_dir, name)
        torch.save({
            "state_dict": ema_state,                 # EMA weights -> what eval loads (released used EMA)
            "model_state_dict_raw": model.state_dict(),
            "ctor_kwargs": ctor,
            "abs_ckpt": {"epoch": len(history["train_loss"])},
            "training_meta": {
                "from_scratch": True, "recipe": "noether-lion", "lr": base_lr,
                "weight_decay": float(cfg["weight_decay"]), "gradient_clip": clip,
                "ema_decay": float(cfg.get("ema_decay", 0.9999)), "epochs": epochs,
                "seed": seed, "params": nparam, "warmup_steps": warmup, "total_steps": total_steps,
            },
        }, path)
        return path

    for ep in range(1, epochs + 1):
        print(f"\n=== epoch {ep}/{epochs} ===", flush=True)
        tr, trp = run_epoch(model, train_loader, dev, scaler, optimizer, scheduler, ema, clip,
                            train=True, max_batches=mtb)
        if args.no_val:
            # no validation: snapshot EMA weights (deepcopy — EMA.state_dict can
            # alias the live shadow tensors) and keep only the last checkpoint
            ema_state = deepcopy(ema.state_dict(model))
            lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr); history["val_loss_ema"].append(None)
            history["val_loss"].append(None); history["lr"].append(lr)
            print(f"epoch {ep}: train {tr:.5f} (p {trp:.5f}) | (no val) | "
                  f"lr {lr:.3e} | {(time.time()-t_start)/60:.1f} min", flush=True)
            save_ckpt("abupt_from_scratch_last.th", ema_state)
        else:
            # validate with EMA weights (faithful to released eval), then restore
            backup = deepcopy(model.state_dict())
            model.load_state_dict(ema.state_dict(model))
            va, vap = run_epoch(model, val_loader, dev, None, None, None, None, clip,
                                train=False, max_batches=mvb)
            ema_state = deepcopy(model.state_dict())
            model.load_state_dict(backup)

            lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr); history["val_loss_ema"].append(va)
            history["val_loss"].append(va); history["lr"].append(lr)
            print(f"epoch {ep}: train {tr:.5f} (p {trp:.5f}) | val(EMA) {va:.5f} (p {vap:.5f}) | "
                  f"lr {lr:.3e} | {(time.time()-t_start)/60:.1f} min", flush=True)

            save_ckpt("abupt_from_scratch_last.th", ema_state)
            if va < best_val:
                best_val, best_epoch = va, ep
                save_ckpt("abupt_from_scratch_best.th", ema_state)
                print(f"  * best (val EMA {va:.5f})", flush=True)
        json.dump({**history, "best_epoch": best_epoch,
                   "best_val": None if best_val == float("inf") else best_val},
                  open(os.path.join(args.save_dir, "train_history.json"), "w"), indent=2)

    if args.no_val:
        print(f"\n[done] {epochs} epochs, no validation | {(time.time()-t_start)/60:.1f} min total")
        print(f"[done] checkpoint in {args.save_dir} (final -> abupt_from_scratch_last.th)")
    else:
        print(f"\n[done] best epoch {best_epoch} val(EMA) {best_val:.5f} | "
              f"{(time.time()-t_start)/60:.1f} min total")
        print(f"[done] checkpoints in {args.save_dir} (best -> abupt_from_scratch_best.th)")


if __name__ == "__main__":
    main()
