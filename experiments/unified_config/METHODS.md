# Unified-configuration baseline — methods

To address the reviewer's concern that Table 5's per-model training configurations differ, we retrained the benchmark models under a single, strictly identical training recipe and report the result as a reference baseline. **Only training hyperparameters are unified; each model keeps its own architecture.**

| knob | value |
|---|---|
| optimizer | adam |
| learning rate | 0.001 |
| weight decay | 0.0 |
| loss | mse |
| scheduler | cosine (T_max=100, eta_min=1e-06) |
| epochs | 100 (no early stopping) |
| batch size | 4 (largest fitting every model on an 80GB A100; see batch_probe_results.json) |
| points per sample | 10000 |
| seed | 42 |

Adam + MSE were chosen so the recipe is enforced with no changes to any model wrapper (the Transolver family hardcodes Adam+MSE; the point-cloud models read both from config). Every model was verified to instantiate Adam + CosineAnnealingLR + MSELoss under this config (`verify_unified.py`).

Results: `unified_config_table.csv` / `.tex`. Significance via `scripts/paired_bootstrap_comparison.py --results-dir results_unified`.

## Note on PointTransformer

PointTransformer **diverged (training loss NaN from epoch 1)** under the unified recipe and is reported as such. The divergence is reproducible and attributable to the strict recipe: its hand-tuned configuration uses lr=1e-4 **with linear LR warmup**, whereas the unified recipe applies lr=1e-3 with no warmup (gradient clipping at 1.0 was active and did not prevent it). All eight other models trained stably (0 NaN epochs). This is itself informative: a single naive training recipe is not stable for every architecture, which is precisely why the main comparison uses per-model best-effort configurations. Among the eight stable models the ranking is preserved and the Transolver family remains best (~0.96 R²).
