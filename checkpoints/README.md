# Pretrained weights

Ten checkpoints, 160 MB in total, shipped in this directory. Nothing to
download.

```bash
python scripts/download_checkpoints.py --verify-only
```

Each file's SHA-256 is checked against `MANIFEST.json`, which also records the
published table row that every checkpoint reproduces.

## Not released

**PointNetLarge.** Its row in Table 1 (Rel L2 0.2436, R2 0.9025) stands, and the
model code and training configuration are in this repository, but the trained
weights are not distributed. Retrain with:

```bash
python scripts/run_single_model.py pointnet_large \
    --dataset-path $CARBENCH_SURFACE_DIR --subset-dir $CARBENCH_SPLITS_DIR
```

## What each checkpoint is

| Name | Reproduces | Rel L2 | R² |
|---|---|---|---|
| `pointnet` | Table 1 | 0.3803 | 0.7639 |
| `fno` | Table 1 | 0.3016 | 0.8503 |
| `pointmae` | Table 1 | 0.2713 | 0.8791 |
| `regdgcnn` | Table 1 | 0.2006 | 0.9327 |
| `point_transformer` | Table 1 | 0.1909 | 0.9359 |
| `tripnet` | Table 1 | 0.1623 | 0.9537 |
| `transolver_plus` | Table 1 | 0.1573 | 0.9543 |
| `transolver` | Table 1 | 0.1503 | 0.9577 |
| `transolver_large` | Table 1 | 0.1457 | 0.9595 |
| `abupt` | Supplementary Table 6 | 0.1539 | 0.9576 |

Relative L2 is the mean over test geometries; R² is pooled over all test points.

## AB-UPT*

The headline AB-UPT row in Table 1 (`AB-UPT*`, Rel L2 0.1358) uses the
checkpoint released by the authors of AB-UPT, not weights trained here, and is
**not redistributed with CarBench**. Obtain it from the AB-UPT authors
(Alkin et al., arXiv:2502.09692) and point CarBench at it:

```bash
export CARBENCH_ABUPT_STAR_CHECKPOINT=/path/to/pressure_abupt_checkpoint.th
```

The `abupt` checkpoint shipped here is our own from-scratch model, trained only
on the official DrivAerNet++ training split.

## TripNet

TripNet fits a triplane occupancy field per geometry before predicting pressure.
The first evaluation of a given design takes roughly 20 s for that fit and is
cached afterwards; the cache is not shipped.
