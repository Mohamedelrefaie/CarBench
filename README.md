# CarBench

**A benchmark for neural surrogates on high-fidelity 3D car aerodynamics.**

Published in **Nature Communications Engineering**
([doi:10.1038/s44172-026-00805-2](https://doi.org/10.1038/s44172-026-00805-2)).

CarBench trains and evaluates eleven neural surrogate models for surface-pressure
prediction on [DrivAerNet++](https://dataverse.harvard.edu/dataverse/DrivAerNet)
under one protocol: the same splits, the same sampling, the same metrics, and a
bootstrap confidence interval on every reported number.

This repository contains the code and the trained weights behind the paper.

- Paper: **Nature Communications Engineering**, 2026 — https://doi.org/10.1038/s44172-026-00805-2
- Leaderboard: https://web.mit.edu/decode/www/carbench/
- Dataset: https://dataverse.harvard.edu/dataverse/DrivAerNet (CC BY-NC 4.0)

## What is here

| | |
|---|---|
| `benchmark/` | the harness: trainer, evaluator, metrics, bootstrap confidence intervals, cross-category evaluation |
| `models/` | the eleven models, each behind one `BenchmarkModel` wrapper |
| `app/` | single-geometry inference and the model registry |
| `scripts/` | train, evaluate, paired bootstrap comparison, cross-category runs, plots |
| `experiments/` | the studies in the paper: full-resolution protocols, sample efficiency, dataset embeddings, from-scratch controls |
| `checkpoints/` | manifest for the released weights (the weights download separately) |

## Install

```bash
git clone https://github.com/Mohamedelrefaie/CarBench.git && cd CarBench
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # or: conda env create -f environment.yml
```

Python 3.12 and PyTorch 2.7 (CUDA 11.8) are what the released results were
produced with.

## Dataset

Download DrivAerNet++ from the Harvard Dataverse, then point CarBench at it:

```bash
export CARBENCH_DATA_ROOT=/path/to/DrivAerNet
```

CarBench expects `$CARBENCH_DATA_ROOT/PressureVTK/Dataverse/*.vtk` and
`$CARBENCH_DATA_ROOT/train_val_test_splits/{train,val,test}_design_ids.txt`.
Every location can be overridden individually; see `carbench_paths.py`.

The official split is used unchanged: **5,819 train / 1,148 validation /
1,154 test** designs, disjoint at the geometry level.

## Weights

Trained weights for ten of the eleven benchmarked models ship with this
repository, in `checkpoints/`. Nothing to download:

```bash
python scripts/download_checkpoints.py --verify-only
```

That checks every file's SHA-256 against `checkpoints/MANIFEST.json`, which also
records the published table row each checkpoint reproduces.

PointNetLarge is the one model whose weights are not released; its code and
configuration are here, so it can be retrained with the standard command below.
AB-UPT* -- the Table 1 headline row -- is the original authors' checkpoint and is
not ours to redistribute; see `checkpoints/README.md`. The `abupt` checkpoint
here is our own from-scratch model.

## Quickstart

Predict the surface-pressure field for one geometry:

```bash
python -m app.smoke_test --model transolver_large --vtk $CARBENCH_SURFACE_DIR/E_S_WW_WM_648.vtk
```

Evaluate one model on the held-out split:

```bash
python scripts/run_single_model.py transolver \
    --dataset-path $CARBENCH_SURFACE_DIR \
    --subset-dir  $CARBENCH_SPLITS_DIR \
    --skip-training
```

Train from scratch:

```bash
python scripts/run_single_model.py transolver \
    --dataset-path $CARBENCH_SURFACE_DIR --subset-dir $CARBENCH_SPLITS_DIR
```

## Reproducing the paper

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for the command behind each table
and figure, and for what each number's aggregation convention is.

Two conventions matter when comparing against the tables:

- **Relative L2** is the mean over test geometries.
- **R²** is pooled over all test points.

They differ, and mixing them changes the numbers. As a check, `MSE / (1 - R²)`
should equal the variance of the targets for every model scored on the same
points.

## Adding a model

Implement the `BenchmarkModel` interface in `benchmark/base_model.py`, drop the
module in `models/<name>/` with a `config.yaml`, and every script here will train,
evaluate and compare it alongside the rest.

## Citation

```bibtex
@article{elrefaie2026carbench,
  title     = {CarBench: A comprehensive benchmark for neural surrogates on
               high-fidelity 3D car aerodynamics},
  author    = {Elrefaie, Mohamed and Shu, Dule and Klenk, Matt and Ahmed, Faez},
  journal   = {Nature Communications Engineering},
  publisher = {Nature Portfolio},
  year      = {2026},
  doi       = {10.1038/s44172-026-00805-2},
  url       = {https://doi.org/10.1038/s44172-026-00805-2}
}
```

Please also cite DrivAerNet++ if you use the dataset.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY-NC 4.0, from the Harvard Dataverse.
The AB-UPT* checkpoint remains under its authors' terms and is not distributed here.
