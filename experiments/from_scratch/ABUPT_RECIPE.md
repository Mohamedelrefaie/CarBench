# AB-UPT from-scratch recipe (extracted from Emmi-AI/noether, DrivAerNet)

Source: `_external/noether/recipes/aero_cfd/` (cloned read-only).
Files: `scripts/train_drivaernet.py`, `presets/drivaernet.py`, `configs/optimizer/lion.yaml`,
`configs/experiment/drivaerml/ab_upt.yaml`, `configs/train_drivaernet.yaml`.

## Architecture (AeroABUPT) → maps to our `models/abupt/surface_only_model.py SurfaceOnlyABUPT`
- hidden_dim/dim = **192**
- geometry_depth = **6**   (our default is 1 → override to 6)
- num_heads = 3 (ViT-tiny)
- physics_blocks = ["perceiver"] + ["shared","cross"]*5  → our `blocks="p"` + `num_surface_blocks=11`
- output_dim_surface = 4 (pressure[1] + wallshear[3]); we train/eval pressure (channel 0)

## Data pipeline (DrivAerNetPreset, AeroABUPT overrides)
- num_geometry_points = **16384**
- num_geometry_supernodes = **1024**
- num_surface_anchor_points = **512**
- num_surface_queries = 0
- num_surface_points = 16384
- surface_position normalizer: strategy="position", **scale=1000** (m→mm-ish)
- surface_pressure normalizer: mean_std (our wrapper uses pressure_mean=-93.41, std=120.79 Pa)

## Optimizer (override /optimizer: lion — the AB-UPT default)
- **Lion**: lr = **5.0e-5**, weight_decay = **0.05**, clip_grad_norm = **0.25**
- schedule: **LinearWarmupCosineDecay**, warmup_percent=0.05, end_value=1e-6, max_value=lr
- (alt Muon+Lion: Muon lr=0.01 wd=0.01 + Lion secondary lr=5e-5 wd=0.05 — optional, not default)

## Budget / misc
- **max_epochs**: hydra `train_drivaernet.yaml` says **50**; the python `train_drivaernet.py`
  hardcodes **2**. AMBIGUOUS → use 50 (canonical config) but verify via a short convergence
  check; flag both in METHODS.
- effective_batch_size = **1**; precision = float16 (mixed precision)
- loss: WeightedLossTrainer, MSE on normalized fields; we keep **surface_pressure only**
  (drop friction + 3 volume fields — our data is surface-only)

## In-harness from-scratch config to write (experiments/from_scratch/configs/abupt.yaml)
- NO checkpoint_path (wrapper logs "will use random initialization" → trains from scratch ✓)
- dim/geometry_depth/num_heads/blocks/num_surface_blocks per arch above
- num_geometry_points=16384, num_geometry_supernodes=1024, num_surface_anchor_points=512
- optimizer: lion, lr: 5.0e-5, weight_decay: 0.05, gradient_clip: 0.25
- scheduler: cosine + warmup (warmup 5%, eta_min 1e-6)
- epochs: 50, batch_size: 1, mixed_precision: true, seed: 42, loss: mse

## RESOLVED from the actual pretrained checkpoint (models/abupt/pressure_abupt_checkpoint.th)
The released checkpoint stores `ctor_kwargs` + `abs_ckpt` — these resolve every ambiguity:
- **Architecture (ctor_kwargs):** `dim=192, geometry_depth=1, blocks='p', num_surface_blocks=11,
  num_heads=3, radius=1, ndim=3, input_dim=3, output_dim_surface=4`.
  → This is the vendored `SurfaceOnlyABUPT()` **DEFAULT** (6,006,532 params). The released
  surface-only checkpoint is geometry_depth=**1**, NOT the train_drivaernet.py demo's
  geometry_depth=6. So for an architecture-matched, leakage-isolating comparison we train the
  IDENTICAL default arch. **No wrapper arch edit needed** (`_setup_model` builds `SurfaceOnlyABUPT()`).
- **Budget (abs_ckpt):** epoch=**30**, update=174480, sample=174480. 174480/30 = 5816 ≈ our
  5819-car train split → confirms batch_size=1 and resolves the 2-vs-50 ambiguity: the real
  released checkpoint trained **30 epochs**. We use 30.
- **EMA:** `ema=0.9999` → the released weights used EMA decay 0.9999. We replicate EMA and
  evaluate the EMA weights.
- **Pretrained baseline metrics (the row we compare against), 1154 test cars:**
  R²=**0.9675**, MSE=559.31 Pa², MAE=**10.81 Pa**, RMSE=**23.65 Pa**, RelL2=**0.1358**
  (from results_backup/surface-abupt-drivaernet/metrics/summary.{md,csv}).

## Comparison design (architecture-matched, isolates pretraining/leakage)
Hold architecture + budget fixed (geometry_depth=1, 30 epochs, batch 1) and the only difference
vs the pretrained row is initialization: pretrained public weights (possible leakage) vs random
init. Optimization mirrors the Noether recipe (Lion + warmup-cosine + the preset point counts).
Trained on the 4 surface channels (pressure[1]+wallshear[3], equal weight, matching FIELD_WEIGHTS
surface_pressure=1.0 + surface_friction=1.0); evaluated on pressure (channel 0) via the EXACT
existing eval path (scripts/train_and_analyze.py abupt --evaluate-only --checkpoint <ours>) with
the same dense eval config (65536 geo / 16384 anchors / 16384 queries) the pretrained row used.

## Env (resolved)
Use `~/miniconda3/envs/paint/bin/python` — the only env with the full stack
(torch 2.5.1+cu124, torch_geometric, pyvista, matplotlib, seaborn, sklearn). Added
`torch_scatter==2.1.2+pt25cu124` (--no-deps) to it; also added psutil+pyparsing to ~/.local.
(paint also powers the protected style_server — additive installs only, never modify deps.)

## Approach (resolved): dedicated trainer, reuse eval path
The abupt wrapper's `forward()` is eval-only (forces model.eval()+no_grad) and its denormalize-in-
forward + DenormalizedMSELoss double-normalize under UnifiedTrainer → NOT usable for training.
So we write a standalone trainer (`experiments/from_scratch/train_abupt.py`) that uses the model +
collator + dataset directly (Lion, warmup-cosine, fp16, grad-clip 0.25, EMA 0.9999), saves a
checkpoint in the released format ({'state_dict':..., 'ctor_kwargs':...}), then evaluates it with
the UNCHANGED eval path so the from-scratch number is computed identically to the pretrained row.
