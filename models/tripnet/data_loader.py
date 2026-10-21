"""Predictor data: serve (fitted triplane, surface query pts in NDC, GT pressure) per car.

Triplanes come from occupancy_fit stage1/stage2 (.pt with planes + bbox). Surface points and
pressure come from the DrivAerNet VTK. Pressure is normalized with the dataset's global
mean/std so denormalized Pa metrics line up with the rest of the benchmark table.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# DrivAerNet surface-pressure normalization (== get_normalization_stats / AB-UPT wrapper)
PRESSURE_MEAN = -93.41
PRESSURE_STD = 120.79


def _read_vtk_surface(vtk_path):
    import pyvista as pv
    m = pv.read(vtk_path)
    pts = np.asarray(m.points, np.float32)
    p = np.asarray(m.point_data["p"], np.float32).reshape(-1)
    return pts, p


class TriplanePressureDataset(Dataset):
    """One car -> (planes 3xCxHxW fp32, query_ndc N x3, pressure_norm N)."""

    def __init__(self, tri_dir, dataset_path, cids, n_query=200_000, train=True, seed=0):
        self.tri_dir = tri_dir
        self.dataset_path = dataset_path
        self.cids = [c for c in cids if os.path.exists(os.path.join(tri_dir, f"{c}.pt"))]
        self.n_query = n_query
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, i):
        cid = self.cids[i]
        tp = torch.load(os.path.join(self.tri_dir, f"{cid}.pt"), map_location="cpu", weights_only=False)
        planes = tp["planes"].float().squeeze(0)            # (3,C,H,W)
        lo = np.asarray(tp["bbox_min"], np.float32)
        hi = np.asarray(tp["bbox_max"], np.float32)
        pts, p = _read_vtk_surface(os.path.join(self.dataset_path, f"{cid}.vtk"))
        n = pts.shape[0]
        if self.n_query and self.n_query < n:
            sel = (self.rng.integers(0, n, self.n_query) if self.train
                   else np.linspace(0, n - 1, self.n_query).astype(np.int64))
            pts, p = pts[sel], p[sel]
        center = (lo + hi) * 0.5
        scale = np.clip((hi - lo) * 0.5, 1e-6, None)
        ndc = ((pts - center) / scale).astype(np.float32)
        p_norm = ((p - PRESSURE_MEAN) / PRESSURE_STD).astype(np.float32)
        return {
            "cid": cid,
            "planes": planes,
            "query_ndc": torch.from_numpy(ndc),
            "pressure": torch.from_numpy(p_norm),
        }


def collate(batch):
    """Stack planes (B,3,C,H,W); keep queries as lists (variable N) -> pad to min N."""
    planes = torch.stack([b["planes"] for b in batch], 0)
    nmin = min(b["query_ndc"].shape[0] for b in batch)
    q = torch.stack([b["query_ndc"][:nmin] for b in batch], 0)
    p = torch.stack([b["pressure"][:nmin] for b in batch], 0)
    return {"planes": planes, "query_ndc": q, "pressure": p, "cids": [b["cid"] for b in batch]}
