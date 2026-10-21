#!/usr/bin/env python
"""Render the test-set dataset-coverage figures from a saved embedding.

    python plot_embedding.py [results_dir]

Produces, in the paper's style:
  pca_datasets_testset.png    PCA   coloured by dataset
  tsne_datasets_testset.png   t-SNE coloured by dataset
  dual_datasets_testset.png   both, side by side (recommended for the paper)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Matches the full-dataset figure (figures/pca_datasets_DrivAerNet-Benchmark.png):
# matplotlib Set2, circles for every dataset, same legend naming. Verified by
# sampling that figure's marker pixels (#66c2a5 / #fc8d62 / #8da0cb).
# dataset -> (display name, colour, marker, z-order)
STYLE = {
    "DrivAerNet": ("DrivAerNet", "#66c2a5", "o", 1),
    "DrivAerML":  ("DrivAerML",  "#fc8d62", "o", 3),
    "SHIFT_SUV":  ("SHIFT_SUV",  "#8da0cb", "o", 2),
}
ORDER = ["DrivAerNet", "SHIFT_SUV", "DrivAerML"]


def _panel(ax, emb, datasets, title, xlabel, ylabel):
    for ds in ORDER:
        m = datasets == ds
        if not m.any():
            continue
        name, colour, marker, z = STYLE[ds]
        ax.scatter(emb[m, 0], emb[m, 1], s=22, c=colour, marker=marker,
                   alpha=0.85, edgecolors="none", zorder=z,
                   label=f"{name} (n={int(m.sum())})")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)


def plot_all(out_dir):
    z = np.load(os.path.join(out_dir, "test_set_embedding.npz"), allow_pickle=True)
    ds = z["datasets"]
    pca, tsne = z["pca"], z["tsne"]
    ev = z["pca_explained"] if "pca_explained" in z.files else [np.nan, np.nan]
    px, py = (f"PC{i+1} ({100*ev[i]:.1f}\\% var)".replace("\\", "") for i in (0, 1))

    for emb, name, title, xl, yl in (
        (pca,  "pca_datasets_testset",  "", "Component 1", "Component 2"),
        (tsne, "tsne_datasets_testset", "", "Component 1", "Component 2"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        _panel(ax, emb, ds, title, xl, yl)
        ax.legend(fontsize=11, frameon=True, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    _panel(axes[0], pca,  ds, "PCA projection",   px, py)
    _panel(axes[1], tsne, ds, "t-SNE projection", "t-SNE 1", "t-SNE 2")
    axes[1].legend(fontsize=11, frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dual_datasets_testset.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, "dual_datasets_testset.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    counts = {d: int((ds == d).sum()) for d in ORDER}
    print(f"figures written to {out_dir}  |  counts: {counts}")


if __name__ == "__main__":
    plot_all(sys.argv[1] if len(sys.argv) > 1 else
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
