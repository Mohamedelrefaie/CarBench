# Reproducing the paper

All commands assume `CARBENCH_DATA_ROOT` is set and the weights are downloaded.

```bash
export CARBENCH_DATA_ROOT=/path/to/DrivAerNet
export CARBENCH_SURFACE_DIR=$CARBENCH_DATA_ROOT/PressureVTK/Dataverse
export CARBENCH_SPLITS_DIR=$CARBENCH_DATA_ROOT/train_val_test_splits
python scripts/download_checkpoints.py --verify-only
```

## Aggregation conventions

| Metric | How it is aggregated |
|---|---|
| Relative L2 | per-geometry ratio, averaged over the 1,154 test designs |
| R² | pooled over all test points |
| MAE, MSE, RMSE | pooled over all test points, in m²·s⁻² |
| P50 … P99 | percentiles of absolute error over all test points |

Consistency check, valid for any model scored on the same points:
`MSE / (1 - R²) = Var(targets)`. Every row of Table 1 implies 14,189–14,306.

## Table 1 — the main leaderboard

```bash
python scripts/evaluate_all_models.py --skip-training
```

Per-model: `python scripts/run_single_model.py <name> --dataset-path $CARBENCH_SURFACE_DIR --subset-dir $CARBENCH_SPLITS_DIR --skip-training`

## Table 2 — cross-category generalization

```bash
python scripts/cross_category_generalization.py       # trains the six splits
python scripts/analyze_cross_category_generalization.py
```

Characterizes Transolver and Transolver++ only, as stated in the paper.

## Table 3 — subsampled, interpolated and patchwise evaluation

```bash
python experiments/full_res_eval/run_full_res_eval.py --strategy patchwise
python experiments/full_res_eval/measure_cost.py
```

## Supplementary Table 3 — shared optimization budget

```bash
python experiments/unified_config/run_unified.py
```

## Supplementary Table 6 — from-scratch AB-UPT and TripNet

```bash
python experiments/from_scratch/eval_abupt.py
bash   experiments/from_scratch/run_tripnet_full.sh
```

## Confidence intervals and significance

```bash
python scripts/paired_bootstrap_comparison.py
```

Stratified paired bootstrap: whole geometries are resampled with replacement
within archetype strata, and the same resamples are used for every model so
comparisons are paired.

## Figures

```bash
python scripts/plot_all_loss_curves.py          # convergence
python experiments/dataset_embedding/embed_test_sets.py
```

## Expected runtime

Evaluation of all eleven models on the full test split is a few GPU-hours;
patchwise full-resolution evaluation adds roughly 4.4 GPU-hours. TripNet fits a
triplane per geometry on first use (~20 s each, cached afterwards).
