# TripNet from-scratch implementation notes (rebuttal point 2)

Re-implementation of **TripNet** (Chen, Elrefaie, Dai, Ahmed — *TripNet: Learning
Large-scale High-fidelity 3D Car Aerodynamics with Triplane Networks*, PoF 2026 /
[arXiv:2503.17400](https://arxiv.org/abs/2503.17400)). **No public code exists**, so this is
implemented from the paper. Goal: a **from-scratch** (no pretrained weights) DrivAerNet
surface-pressure number, comparable to the pretrained TripNet row already in our tables.

Evaluated on **our exact DrivAerNet test split + metrics** via the benchmark harness, so the
result lines up with the table. Trained on **surface pressure only** (our data is surface-only).

This file documents (a) what the paper specifies and (b) **every architectural/training choice
the paper leaves open** that we had to fix — required for an honest rebuttal.

---

## 1. Triplane representation (paper-specified)
- 3 axis-aligned feature planes (xy, yz, xz), each **C=32** channels at **128×128**.
- A 3D query point p=(x,y,z), with car coords normalized to [-1,1]^3 by the car bbox, is
  orthographically projected onto each plane and **bilinearly sampled** (`grid_sample`),
  giving 3×32 features → concatenated to **96-d** per point.

## 2. Geometry encoder = occupancy auto-decoding (paper-specified, 2-stage)
The triplane is a *learned* per-car geometry code, fit so a shared decoder reproduces the car's
**occupancy** field.
- **Stage 1 (joint):** learnable triplanes for 500 cars + a shared occupancy decoder MLP, trained
  jointly. **500 epochs, lr 1e-3**, sampling **5M uniform + 5M near-surface** points/car.
- **Stage 2 (per-car):** freeze the shared decoder; for every other car optimize ONLY its triplane
  (random init) for **70 epochs (~20 s/car)**. Cache the fitted 3×32×128×128 triplane to disk.

### Occupancy labels — DATA ADAPTATION (paper assumes watertight geometry; ours is not)
The DrivAerNet **CFD surface meshes here are not watertight** (is_watertight=False, vol≈0) and no
STL geometries are on disk. We label occupancy **directly on the existing triangulated CFD mesh**
via open3d `RaycastingScene.compute_occupancy` (ray-parity), which is robust to the small holes:
- Probed open3d **Poisson** reconstruction first — correct but **~212 s/car** (×8156 ≈ 20 CPU-days):
  rejected as far too slow.
- Direct raw-mesh ray-parity occupancy: scene build ~0.7 s + 500k-pt occupancy ~1.0 s **per car**,
  with sane inside fractions (uniform ≈0.45, near-surface ≈0.46 on 3 cars). This is both faster
  AND more faithful (uses the actual CFD geometry, no reconstruction error). All-car labeling is
  thus ~hours, not weeks.
- This is still a deviation from the paper (which uses native watertight geometry + true occupancy);
  it only affects the *geometry code*, not the eval target. The slight non-watertightness adds a
  little label noise at open regions (underbody/wheels), absorbed by the auto-decoded triplane.

## 3. Pressure predictor (paper: shared U-Net + bilinear query + 4-layer MLP)
- Input: the car's **frozen, fitted** triplane (3×32×128×128).
- A **shared 2D U-Net** refines the triplane (we concat 3×32→96 channels, U-Net 96→96, split back
  to 3×32 processed planes).
- For each surface query point: bilinear-sample the 3 processed planes → concat 96 → **4-layer MLP**
  → scalar pressure.
- Training: **200 epochs, Adam lr 1e-3, batch 8, 200k query pts/car, MSE** on normalized pressure.

## 4. Choices the paper does NOT specify (fixed here, documented)
| component | paper | our choice (documented) |
|---|---|---|
| U-Net depth/width | "shared U-Net" only | 4-level encoder/decoder, base 64ch (96→64→128→256→512), skip connections, GroupNorm+SiLU |
| triplane→U-Net layout | unspecified | concat 3 planes channel-wise (96ch on 128×128), U-Net 96→96, split to 3×32 |
| MLP width | "4-layer MLP" | width 256, SiLU; in 96 → 256 → 256 → 256 → 1 |
| occupancy decoder | unspecified | 4-layer MLP width 128, in 96 → 1 logit, BCE-with-logits |
| coord normalization | unspecified | per-car bbox → [-1,1]^3 (consistent for fit + query) |
| near-surface sampling | "near-surface" | surface samples + N(0, σ) jitter, σ = 1% of bbox diagonal |
| pressure normalization | — | global PRESSURE_MEAN/STD (same as the benchmark) so Pa metrics match the table |
| optimizer (fit) | lr 1e-3 | Adam, MSE/BCE; triplanes init N(0, 0.01) |

## 5. Scope / status
Per author direction: **validate the full pipeline on ~50 cars first** (occupancy + reconstruction
sanity, per-car fit time), report the extrapolated full-run cost (8,156 cars), then green-light the
full run. Files: `triplane.py`, `occupancy_fit.py`, `unet.py`, `model.py`, `data_loader.py`,
`wrapper.py`, `config.yaml`. Triplane cache + checkpoints are gitignored; summary CSVs kept.

## Validation (per author direction: validate small, then decide)

### Occupancy / reconstruction (3-car + 60-car)
- Raw-mesh ray-parity occupancy: inside-fraction steady ≈0.44 across 60 cars (sane). Poisson
  rejected (~212 s/car). Direct labeling ≈3.8 s/car (full 1M+1M pool).
- Auto-decoding **fits occupancy nearly perfectly**: stage1 BCE 0.69→0.03 (35 cars, 150 ep);
  stage2 per-car BCE ≈0.03–0.04. BCE plateaus by ~epoch 60 (stage1) → epochs can be cut.

### 50-car end-to-end validation (mechanics + learning, NOT a final number)
Train 50 cars / 60 ep / 200k query, eval 10 held-out test cars:
- normalized: relL2 **0.346**, relL1 0.353, MSE 0.125, MAE 0.215
- Pascal: **R² 0.880**, MAE 25.9, RMSE 42.6, relL2 0.277
- paper (pretrained, full train set, 200 ep): relL2 0.2035, MSE 0.0423, MAE 0.1110
With only 50 training cars the predictor already reaches R²=0.88 — pipeline learns pressure and
gives sensible numbers; the full train set + 200 ep should approach the paper's relL2≈0.20.

### Measured per-unit cost → full-run extrapolation (8,156 cars)
| stage | measured | full-run (paper epochs) | parallel |
|---|---|---|---|
| prep | 3.8 s/car | 8.6 h serial | yes (CPU per-car) → ~0.5–1 h |
| stage1 (joint 500×500ep) | 0.199 s/(car·ep) | ~13.8 h | no (single joint job) |
| stage2 (70ep × 7,656) | 14.65 s/car | ~31 h | yes (per-car, multi-GPU) |
| predictor (200ep × 5,818) | ~0.32 s/step (bs8) | ~13 h | partial |

≈ **58 GPU-h** at paper epochs. Reduced (stage1 150 / stage2 40 / predictor 150 ep — justified by
early BCE/MSE plateau) ≈ **28 GPU-h**. Triplane cache: 3,156 KB/car ×8,156 ≈ **24 GB** (fp16).
