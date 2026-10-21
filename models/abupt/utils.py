import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_pointcloud_single(
    pos,
    color=None,
    title=None,
    alpha=0.5,
    num_points=10000,
    figsize=(6, 6),
):
    perm = torch.randperm(len(pos), generator=torch.Generator().manual_seed(0))[:num_points]
    plt.close()
    plt.clf()
    view_rotation = [20, 125, 0]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = pos[perm].unbind(-1)
    if color is None:
        scatter = ax.scatter(x, y, z, s=3, c="k", alpha=alpha)
    else:
        scatter = ax.scatter(x, y, z, s=3, c=color[perm], cmap="coolwarm", alpha=alpha)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.axis("equal")
    ax.view_init(*view_rotation)
    if title is not None:
        ax.set_title(title)
    if color is not None:
        plt.colorbar(scatter, orientation="horizontal")
    plt.show()


def plot_pointcloud_double(
    pos,
    color,
    title=None,
    alpha=0.5,
    num_points=10000,
    figsize=(18, 6),
    delta_clamp=None,
    view_rotation = (20, 125, 0),
):
    perm = torch.randperm(len(pos[0]), generator=torch.Generator().manual_seed(0))[:num_points]
    plt.close()
    plt.clf()

    fig = plt.figure(figsize=figsize)
    axs = []
    vmin = None
    vmax = None
    delta = color[1] - color[0]
    if delta_clamp is not None:
        delta = delta.clamp(*delta_clamp)
    scatters = []
    for i in range(len(pos) + 1):
        is_delta = i == 2
        ax = fig.add_subplot(130 + i + 1, projection="3d")
        axs.append(ax)
        if is_delta:
            x, y, z = pos[0][perm].unbind(-1)
        else:
            x, y, z = pos[i][perm].unbind(-1)
        if vmin is None:
            vmin = color[i].min()
            vmax = color[i].max()
        scatter = ax.scatter(
            x,
            y,
            z,
            s=3,
            c=delta[perm] if is_delta else color[i][perm],
            cmap="coolwarm",
            vmin=None if is_delta else vmin,
            vmax=None if is_delta else vmax,
            alpha=alpha,
        )
        scatters.append(scatter)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        plt.axis("equal")
        ax.view_init(*view_rotation)
        if title is not None:
            if is_delta:
                ax.set_title("delta")
            else:
                ax.set_title(title[i])
    fig.colorbar(scatters[0], ax=axs[:2], orientation="horizontal")
    fig.colorbar(scatters[2], ax=axs[-1], orientation="horizontal")
    plt.show()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    plot_pointcloud_double(
        pos=[torch.randn(100, 3), torch.randn(100, 3)],
        color=[torch.randn(100), torch.randn(100)],
        title=["target", "prediction"],
    )
