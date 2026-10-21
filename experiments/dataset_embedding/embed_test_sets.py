#!/usr/bin/env python
"""Dataset-coverage embedding (PCA + t-SNE) computed on TEST SETS only.

Reproduces the methodology of DrivAerNet_Analysis/tsne_all_fixed.py — sample a
point cloud per STL, train the same small PointNet on the car-type labels, embed
its penultimate features — but restricted to held-out geometries:

  DrivAerNet++ : the official test split (design IDs read from the split file)
  DrivAerML    : --n-drivaerml geometries (default 50)
  SHIFT-SUV    : --frac-shiftsuv of the dataset (default 0.10)

Selections are seeded and written next to the figures so the figure is
reproducible. Both projections are produced from the same features.

    python embed_test_sets.py --gpu 1
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

DATA = "/mnt/decodenas2/mohamed/datasets"
PATHS = {
    "DrivAerNet": f"{DATA}/DrivAerNet/all_stls",
    "DrivAerML": f"{DATA}/FoundationEngineeringModel/drivaerml",
    "SHIFT_SUV": f"{DATA}/FoundationEngineeringModel/shiftsuv",
}
TEST_IDS = os.path.join(os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "train_val_test_splits")), "test_design_ids.txt")
HERE = os.path.dirname(os.path.abspath(__file__))

# same label rule as the original pipeline
def label_of(stem: str, dataset: str) -> str:
    if dataset == "DrivAerNet":
        return {"N": "notchback", "F": "fastback", "E": "estateback"}.get(stem[0], "unknown")
    if dataset == "SHIFT_SUV":
        s = stem.lower()
        return "estateback" if "estate" in s else "fastback" if "fastback" in s else "unknown"
    if dataset == "DrivAerML":
        return "notchback"
    return "unknown"


def select_files(args):
    """Deterministic, seeded selection of the test-set geometries per dataset."""
    rng = random.Random(args.seed)
    sel = {}

    test_ids = {l.strip() for l in open(TEST_IDS) if l.strip()}
    p = Path(PATHS["DrivAerNet"])
    dn = sorted(f for f in p.glob("*.stl") if f.stem in test_ids)
    sel["DrivAerNet"] = dn

    ml = sorted(Path(PATHS["DrivAerML"]).glob("*.stl"))
    sel["DrivAerML"] = sorted(rng.sample(ml, min(args.n_drivaerml, len(ml))))

    sv = sorted(Path(PATHS["SHIFT_SUV"]).glob("*.stl"))
    k = max(1, int(round(args.frac_shiftsuv * len(sv))))
    sel["SHIFT_SUV"] = sorted(rng.sample(sv, k))

    if args.max_per_dataset:
        sel = {k: v[:args.max_per_dataset] for k, v in sel.items()}
    for name, files in sel.items():
        print(f"  {name:11s}: {len(files):5d} geometries selected", flush=True)
    return sel


def load_pointcloud(path, n_points, cache_dir):
    import trimesh
    cf = os.path.join(cache_dir, f"{Path(path).stem}_{n_points}.npy")
    if os.path.exists(cf):
        try:
            return np.load(cf)
        except Exception:
            pass
    mesh = trimesh.load(str(path), process=False, force="mesh")
    pts = np.asarray(mesh.sample(n_points), dtype=np.float32)
    pts -= pts.mean(0)                       # centre
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale                         # unit sphere
    np.save(cf, pts)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--n-drivaerml", type=int, default=50)
    ap.add_argument("--frac-shiftsuv", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-per-dataset", type=int, default=None,
                    help="debug: cap geometries per dataset (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    assert args.gpu != "0", "GPU 0 is reserved"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import torch.nn as nn
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "pc_cache"); os.makedirs(cache, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    print("selecting test geometries", flush=True)
    sel = select_files(args)
    files, datasets, cartypes = [], [], []
    for ds, fl in sel.items():
        for f in fl:
            lab = label_of(f.stem, ds)
            if lab != "unknown":
                files.append(f); datasets.append(ds); cartypes.append(lab)
    print(f"total {len(files)} geometries", flush=True)

    # ---- point clouds ----
    # Processes, not threads: parsing a SHIFT-SUV STL (~15M vertices, 250 MB) is
    # CPU-bound inside trimesh, so a thread pool saturates at ~2.5 cores on the
    # GIL (measured: 250% CPU, 0 MB/s read) no matter how many workers are used.
    # Results are cached per geometry, so a restart resumes rather than redoes.
    # Forking here is safe: CUDA is not initialised until the model moves to GPU.
    print(f"loading point clouds ({args.workers} processes)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        clouds = list(ex.map(partial(load_pointcloud, n_points=args.n_points,
                                     cache_dir=cache), files, chunksize=1))
    X = np.stack(clouds).astype(np.float32)
    print(f"point clouds: {X.shape}", flush=True)

    # ---- same small PointNet, trained on car-type labels ----
    classes = sorted(set(cartypes))
    y = np.array([classes.index(c) for c in cartypes], dtype=np.int64)

    class PointNet(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.feat = nn.Sequential(
                nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(512, n_cls))

        def embed(self, x):                       # x: [B, N, 3]
            return self.feat(x.transpose(1, 2)).max(dim=2)[0]   # [B, 1024]

        def forward(self, x):
            return self.head(self.embed(x))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = PointNet(len(classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    n, bs = len(Xt), 32
    net.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = corr = 0; run = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:                       # BatchNorm needs >1
                continue
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad(); out = net(xb); loss = lossf(out, yb)
            loss.backward(); opt.step()
            run += float(loss) * len(idx); tot += len(idx)
            corr += int((out.argmax(1) == yb).sum())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{args.epochs}  loss {run/tot:.4f}  acc {corr/tot:.3f}", flush=True)

    # ---- features ----
    net.eval(); feats = []
    with torch.no_grad():
        for i in range(0, n, 64):
            feats.append(net.embed(Xt[i:i + 64].to(dev)).cpu().numpy())
    F = np.concatenate(feats)
    Fs = StandardScaler().fit_transform(F)
    print(f"features: {F.shape}", flush=True)

    pca = PCA(n_components=2, random_state=args.seed)
    emb_pca = pca.fit_transform(Fs)
    emb_tsne = TSNE(n_components=2, perplexity=min(args.perplexity, (len(Fs) - 1) / 3),
                    max_iter=1000, random_state=args.seed, init="pca").fit_transform(Fs)

    np.savez(os.path.join(args.out_dir, "test_set_embedding.npz"),
             features=F, pca=emb_pca, tsne=emb_tsne,
             datasets=np.array(datasets), car_types=np.array(cartypes),
             files=np.array([str(f) for f in files]),
             pca_explained=pca.explained_variance_ratio_)
    json.dump({ds: [Path(f).name for f in fl] for ds, fl in sel.items()},
              open(os.path.join(args.out_dir, "selected_geometries.json"), "w"), indent=1)
    print("saved embedding + selection", flush=True)

    from plot_embedding import plot_all          # local module
    plot_all(args.out_dir)


if __name__ == "__main__":
    main()
