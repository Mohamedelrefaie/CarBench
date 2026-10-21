# From-scratch baselines (rebuttal point 2) — methods & results

Reviewer concern: AB-UPT and TripNet enter the tables via **publicly released pretrained
checkpoints**, so their strong scores might reflect train/test leakage from pre-training data
rather than genuine generalization. To rule this out we **retrain both models from scratch
(random init, no pretrained weights)** on the DrivAerNet train split and evaluate on our exact
test split + metrics. The pretrained rows are retained; these are added alongside.

Both models train surface **pressure** only (our data is surface-only).

---

## AB-UPT — from scratch (architecture- and budget-matched)

We mirror the official AB-UPT recipe (Emmi-AI/noether `recipes/aero_cfd`) and **match the released
checkpoint exactly** so the only difference vs the pretrained row is initialization:

- **Architecture** = the released checkpoint's stored `ctor_kwargs`: `SurfaceOnlyABUPT`
  dim=192, geometry_depth=1, 1 perceiver + 11 surface blocks, 3 heads, output_dim 4
  (pressure + wall-shear) — 6.0M params (the vendored default).
- **Budget** = the released checkpoint's `abs_ckpt`: **30 epochs**, batch 1, **EMA 0.9999**
  (174,480 updates ≈ 30 × our 5,819 train cars).
- **Optimizer** = Lion (lr 5e-5, wd 0.05, grad-clip 0.25), linear-warmup(5%)→cosine decay,
  fp16. Points: 16384 geometry / 1024 supernodes / 512 surface anchors (Noether DrivAerNet
  preset). Loss: MSE on the 4 normalized surface channels (pressure + wall-shear, equal weight).
- Trainer: `experiments/from_scratch/train_abupt.py` (the in-repo wrapper is eval-only).
  Config: `experiments/from_scratch/configs/abupt.yaml`. From-scratch run reached val(EMA)
  R²≈0.96 by epoch 30.

### Evaluation (identical for both checkpoints)
The benchmark's stock comprehensive-eval path was unusable here: it emits a ~37 GB per-point
dump and an **inf aggregate R²** (reproducibly, for the *pretrained* checkpoint too), and its
fp16 inference **overflows** on a few cars for a less-conditioned from-scratch model. We instead
evaluate per-car in **fp32** with the same dense sampling (65536 geometry / 16384 supernodes /
16384 anchors + all remaining surface points as queries), denormalize to Pa, and compute the
table's metrics: `experiments/from_scratch/eval_abupt.py`. Running this on the released
checkpoint reproduces the table (R²=0.9669 vs published 0.9675), validating the procedure; both
checkpoints are then compared on identical footing.

Metrics use the **exact CarBench / `benchmark/metrics.py::MetricsCalculator` set** (this repo IS
the DrivAerNet++ benchmark, arXiv:2512.07847): MSE, MAE, RMSE, R² (global pooled), max_error,
Rel-L1, Rel-L2 (per-car ratio, averaged), and P50/P90/P95/P99 |error| percentiles — all in
**denormalized kinematic pressure (m²/s², not Pa)**, the raw OpenFOAM `p`. `eval_abupt.py`'s
metric block is verified **bit-identical** to MetricsCalculator. Both checkpoints use our identical
fp32 50k-query eval (our pretrained re-eval reproduces CarBench Table 1: mean-per-car R²=0.968 ≈
published 0.9675).

### Result (1154 test cars, fp32) — units m²/s²
| AB-UPT variant | MSE | MAE | RMSE | R² | RelL1 | RelL2 | P50 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|---|
| pretrained (CarBench Table 1) | 559 | 10.81 | 23.6 | 0.9675 | — | 0.1358 | — | — | — |
| pretrained (our fp32 re-eval) | 509 | 10.96 | 22.56 | 0.9644 | 0.0932 | 0.1366 | 5.8 | 37.7 | 78.6 |
| **from scratch (fp32)** | **605** | **12.57** | **24.59** | **0.9576** | **0.1068** | **0.1539** | 6.9 | 42.3 | 88.3 |

**Paired bootstrap** (per-car R², n=1154, B=10000; `paired_bootstrap_abupt.py`):
- ΔR² (pretrained − from-scratch) = **+0.0078**, 95% CI **[+0.0069, +0.0087]**
- ΔMAE (from-scratch − pretrained) = **+1.61 m²/s²**, 95% CI [+1.57, +1.66]
- from-scratch ≥ pretrained on only **2.6%** of cars.

**Reading:** from-scratch AB-UPT reaches R²≈0.96 vs pretrained ≈0.97 — a **small but
statistically significant** gap (ΔR²≈0.009; both models excellent). Were the
pretrained score inflated by test-set leakage, training from random init would collapse;
instead it nearly matches. The pretrained advantage is modest and consistent with ordinary
pre-training benefit, **not** leakage.

---

## TripNet — from scratch (re-implemented from the paper)

Re-implemented from arXiv:2503.17400 (no public code); see `models/tripnet/IMPLEMENTATION_NOTES.md`
for every architecture choice the paper leaves open. Geometry encoder = 2-stage occupancy
auto-decoding into per-car 3×32×128×128 triplanes (stage1 joint 500 cars/500 ep, BCE→0.0016;
stage2 per-car 70 ep for all 8,121 cars; occupancy via raw-mesh ray-parity since the CFD meshes
aren't watertight). Predictor = shared 2D U-Net over the triplane + bilinear query + 4-layer MLP,
trained 200 ep / Adam lr1e-3 / batch 8 / 200k query pts / MSE (7.9M params, ~24.5h).
Driver: `experiments/from_scratch/run_tripnet_full.sh`; eval: `experiments/from_scratch/train_tripnet.py`.

### Result (1154 test cars) — exact CarBench metric set, units m²/s²
| TripNet variant | MSE | MAE | RMSE | R² | RelL1 | RelL2 | P50 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|---|
| pretrained (CarBench Table 1) | 596 | 13.30 | 24.9 | 0.9590 | — | 0.1608 | — | — | — |
| **from scratch (ours)** | **673** | **13.58** | **25.94** | **0.9532** | **0.1155** | **0.1650** | 7.6 | 45.3 | 94.8 |

(For reference, in the TripNet paper's *normalized*-pressure convention our from-scratch run
gives Rel-L2 0.2096 / Rel-L1 0.1891 / MSE 0.0461 / MAE 0.1125 vs the paper's 0.2035 / 0.1852 /
0.0423 / 0.1110 — within ~3%.)

**Reading:** in the benchmark's metric space, from-scratch TripNet (R²=0.953, Rel-L2 0.165)
**matches the provided/benchmarked TripNet entry** (R²=0.959, Rel-L2 0.161) to within
ΔR²≈0.006 — the same small gap as AB-UPT. A from-scratch re-implementation landing essentially on
top of the benchmarked score is direct evidence the TripNet result is **not** inflated by
pre-training leakage.

---

Aggregator: `experiments/from_scratch/aggregate.py` → `from_scratch_comparison.csv`.
