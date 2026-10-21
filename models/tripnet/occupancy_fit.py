"""Triplane geometry encoder for TripNet via 2-stage occupancy auto-decoding.

Pipeline (arXiv:2503.17400, adapted — see IMPLEMENTATION_NOTES.md):
  prep:   per car, ray-parity occupancy pool (uniform + near-surface) cached to .npz
  stage1: jointly fit triplanes for N1 cars + a shared occupancy decoder (BCE)
  stage2: freeze decoder, fit each remaining car's triplane; cache 3x32x128x128 + bbox

Run:
  python occupancy_fit.py prep   --cars-file <ids> --out <pooldir> [--n-uniform .. --n-surface ..]
  python occupancy_fit.py stage1 --pooldir <d> --cars-file <ids1> --out <ckptdir> [--epochs ..]
  python occupancy_fit.py stage2 --pooldir <d> --cars-file <ids2> --decoder <ckpt> --out <tridir>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triplane import TriplaneParam, sample_triplane  # noqa: E402
from unet import OccupancyDecoder                     # noqa: E402

CHANNELS, RES = 32, 128


# --------------------------- occupancy pools -------------------------------
def build_scene(vtk_path):
    import open3d as o3d
    import pyvista as pv
    m = pv.read(vtk_path).triangulate()
    v = np.asarray(m.points, np.float32)
    fc = np.asarray(m.faces).reshape(-1, 4)[:, 1:].astype(np.uint32)
    tm = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(fc))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    return scene, v


def make_pool(vtk_path, n_uniform, n_surface, pad=0.03, sigma_frac=0.01, seed=0):
    """Return (pts_ndc float32 [M,3], occ float32 [M], bbox_min[3], bbox_max[3])."""
    import open3d as o3d
    rng = np.random.default_rng(seed)
    scene, v = build_scene(vtk_path)
    lo, hi = v.min(0), v.max(0)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))
    # uniform in padded bbox
    qu = rng.uniform(lo - pad * ext, hi + pad * ext, size=(n_uniform, 3)).astype(np.float32)
    # near-surface: surface verts + gaussian jitter
    idx = rng.integers(0, len(v), n_surface)
    qs = (v[idx] + rng.standard_normal((n_surface, 3)).astype(np.float32) * sigma_frac * diag)
    pts = np.concatenate([qu, qs], 0)
    occ = scene.compute_occupancy(o3d.core.Tensor(pts), nthreads=8).numpy().astype(np.float32)
    center = (lo + hi) * 0.5
    scale = np.clip((hi - lo) * 0.5, 1e-6, None)
    pts_ndc = ((pts - center) / scale).astype(np.float32)
    return pts_ndc, occ, lo.astype(np.float32), hi.astype(np.float32)


def cmd_prep(args):
    cars = _read_ids(args.cars_file, args.dataset_path, args.limit)
    if getattr(args, "shard", None):  # "i/n" -> handle cars where idx%n==i (parallel CPU prep)
        si, sn = (int(x) for x in args.shard.split("/"))
        cars = [pr for k, pr in enumerate(cars) if k % sn == si]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for i, (cid, path) in enumerate(cars):
        outp = os.path.join(args.out, f"{cid}.npz")
        if os.path.exists(outp) and not args.overwrite:
            continue
        pts, occ, lo, hi = make_pool(path, args.n_uniform, args.n_surface, seed=i)
        np.savez_compressed(outp, pts_ndc=pts, occ=occ, bbox_min=lo, bbox_max=hi)
        if i % 10 == 0:
            print(f"  prep {i+1}/{len(cars)} {cid} inside={occ.mean():.3f} "
                  f"{(time.time()-t0)/(i+1):.2f}s/car", flush=True)
    print(f"[prep] {len(cars)} cars -> {args.out} in {(time.time()-t0)/60:.1f} min", flush=True)


# ------------------------------ fitting ------------------------------------
def _load_pool(pooldir, cid, device, n=None):
    z = np.load(os.path.join(pooldir, f"{cid}.npz"))
    pts = torch.from_numpy(z["pts_ndc"]).to(device)
    occ = torch.from_numpy(z["occ"]).to(device)
    if n is not None and n < pts.shape[0]:
        sel = torch.randint(0, pts.shape[0], (n,), device=device)
        pts, occ = pts[sel], occ[sel]
    return pts, occ, z["bbox_min"], z["bbox_max"]


def cmd_stage1(args):
    device = "cuda"
    cids = [c for c, _ in _read_ids(args.cars_file, args.dataset_path, args.limit)]
    cids = [c for c in cids if os.path.exists(os.path.join(args.pooldir, f"{c}.npz"))]
    os.makedirs(args.out, exist_ok=True)
    decoder = OccupancyDecoder(in_dim=3 * CHANNELS).to(device)
    tris = {c: TriplaneParam(CHANNELS, RES).to(device) for c in cids}
    params = list(decoder.parameters())
    for t in tris.values():
        params += list(t.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    # resume from a periodic checkpoint if present
    start_ep = 1
    ckpt = os.path.join(args.out, "stage1_ckpt.pt")
    if getattr(args, "resume", False) and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=device)
        decoder.load_state_dict(st["decoder"])
        for c in cids:
            if c in st["tris"]:
                tris[c].planes.data.copy_(st["tris"][c].to(device))
        opt.load_state_dict(st["opt"]); start_ep = st["epoch"] + 1
        print(f"  stage1 resumed from epoch {st['epoch']}", flush=True)
    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        order = np.random.permutation(cids)
        tot = 0.0
        for c in order:
            pts, occ, _, _ = _load_pool(args.pooldir, c, device, n=args.batch_pts)
            feat = tris[c](pts.unsqueeze(0)).squeeze(0)
            loss = bce(decoder(feat), occ)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss)
        if ep % max(1, args.epochs // 20) == 0 or ep == 1:
            print(f"  stage1 ep {ep}/{args.epochs} bce {tot/len(cids):.4f} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        if ep % args.ckpt_every == 0 and ep < args.epochs:
            torch.save({"epoch": ep, "decoder": decoder.state_dict(), "opt": opt.state_dict(),
                        "tris": {c: tris[c].planes.detach().cpu() for c in cids}}, ckpt)
    torch.save({"decoder": decoder.state_dict(), "channels": CHANNELS, "res": RES},
               os.path.join(args.out, "decoder.pt"))
    for c in cids:
        z = np.load(os.path.join(args.pooldir, f"{c}.npz"))
        _save_triplane(args.out, c, tris[c], z["bbox_min"], z["bbox_max"])
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"[stage1] {len(cids)} cars, {args.epochs} ep -> {args.out} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


def cmd_stage2(args):
    device = "cuda"
    pairs = _read_ids(args.cars_file, args.dataset_path, args.limit)
    if not args.on_the_fly:
        pairs = [(c, p) for c, p in pairs if os.path.exists(os.path.join(args.pooldir, f"{c}.npz"))]
    if args.shard:  # "i/n" -> this proc handles cars where idx % n == i
        i, n = (int(x) for x in args.shard.split("/"))
        pairs = [pr for k, pr in enumerate(pairs) if k % n == i]
    os.makedirs(args.out, exist_ok=True)
    dec_ck = torch.load(args.decoder, map_location=device)
    decoder = OccupancyDecoder(in_dim=3 * CHANNELS).to(device)
    decoder.load_state_dict(dec_ck["decoder"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    bce = torch.nn.BCEWithLogitsLoss()
    t0 = time.time(); done = 0
    for i, (c, vtk) in enumerate(pairs):
        outp = os.path.join(args.out, f"{c}.pt")
        if os.path.exists(outp) and not args.overwrite:
            continue
        if args.on_the_fly:
            pool_pts, pool_occ, lo, hi = gen_pool_inmem(vtk, args.n_uniform, args.n_surface, device, seed=i)
        else:
            pool_pts, pool_occ, lo, hi = _load_pool(args.pooldir, c, device)
        tri = TriplaneParam(CHANNELS, RES).to(device)
        opt = torch.optim.Adam(tri.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            sel = torch.randint(0, pool_pts.shape[0], (args.batch_pts,), device=device)
            feat = tri(pool_pts[sel].unsqueeze(0)).squeeze(0)
            loss = bce(decoder(feat), pool_occ[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        _save_triplane(args.out, c, tri, lo, hi, final_bce=float(loss))
        done += 1
        if done % 25 == 1:
            print(f"  stage2 {i+1}/{len(pairs)} {c} bce {float(loss):.4f} "
                  f"{(time.time()-t0)/done:.2f}s/car", flush=True)
    print(f"[stage2] shard={args.shard or 'all'} {done} cars fit, {args.epochs} ep -> {args.out} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


def _save_triplane(out, cid, tri, bbox_min, bbox_max, final_bce=None):
    torch.save({"planes": tri.planes.detach().half().cpu(),  # 3x32x128x128 (fp16 to save disk)
                "bbox_min": np.asarray(bbox_min, np.float32),
                "bbox_max": np.asarray(bbox_max, np.float32),
                "final_bce": final_bce},
               os.path.join(out, f"{cid}.pt"))


def gen_pool_inmem(vtk_path, n_uniform, n_surface, device, seed=0):
    """Generate an occupancy pool in memory (no disk) -> tensors on device + bbox."""
    pts, occ, lo, hi = make_pool(vtk_path, n_uniform, n_surface, seed=seed)
    return (torch.from_numpy(pts).to(device), torch.from_numpy(occ).to(device), lo, hi)


# ------------------------------ helpers ------------------------------------
def _read_ids(cars_file, dataset_path, limit):
    with open(cars_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    if limit:
        ids = ids[:limit]
    out = []
    for cid in ids:
        p = os.path.join(dataset_path, f"{cid}.vtk")
        if os.path.exists(p):
            out.append((cid, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict()
    for name in ("prep", "stage1", "stage2"):
        p = sub.add_parser(name)
        p.add_argument("--dataset-path", default=os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "PressureVTK", "Dataverse")))
        p.add_argument("--cars-file", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--overwrite", action="store_true")
        if name == "prep":
            p.add_argument("--n-uniform", type=int, default=1_000_000)
            p.add_argument("--n-surface", type=int, default=1_000_000)
            p.add_argument("--shard", default=None, help="i/n -> handle cars where idx%%n==i")
        else:
            p.add_argument("--epochs", type=int, default=(500 if name == "stage1" else 70))
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--batch-pts", type=int, default=200_000)
        if name == "stage1":
            p.add_argument("--pooldir", required=True)
            p.add_argument("--ckpt-every", type=int, default=50)
            p.add_argument("--resume", action="store_true")
        if name == "stage2":
            p.add_argument("--decoder", required=True)
            p.add_argument("--pooldir", default=None)
            p.add_argument("--on-the-fly", action="store_true",
                           help="generate occupancy pool in memory per car (no pool cache)")
            p.add_argument("--n-uniform", type=int, default=1_000_000)
            p.add_argument("--n-surface", type=int, default=1_000_000)
            p.add_argument("--shard", default=None, help="i/n -> handle cars where idx%%n==i")
    args = ap.parse_args()
    {"prep": cmd_prep, "stage1": cmd_stage1, "stage2": cmd_stage2}[args.cmd](args)


if __name__ == "__main__":
    main()
