#!/usr/bin/env python
"""Verify the CarBench pretrained weights against the manifest.

The weights ship with the repository in checkpoints/, so normally there is
nothing to fetch and this just checks every file's SHA-256 -- a truncated or
substituted checkpoint is caught before you evaluate with it. Setting
CARBENCH_WEIGHTS_URL additionally allows fetching a missing file from a mirror.

    python scripts/download_checkpoints.py                 # all models
    python scripts/download_checkpoints.py transolver fno  # a subset
    python scripts/download_checkpoints.py --verify-only   # re-check local files
"""
import argparse, hashlib, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.environ.get("CARBENCH_CHECKPOINT_DIR", os.path.join(HERE, "checkpoints"))
MANIFEST = os.path.join(CKPT_DIR, "MANIFEST.json")
# set at release time; see checkpoints/README.md
BASE_URL = os.environ.get("CARBENCH_WEIGHTS_URL", "")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model names (default: all)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    man = json.load(open(MANIFEST))["models"]
    names = a.models or sorted(man)
    bad = 0
    for n in names:
        if n not in man:
            print(f"  {n}: unknown model (known: {', '.join(sorted(man))})"); bad += 1; continue
        meta = man[n]
        dst = os.path.join(CKPT_DIR, meta["file"])
        if not os.path.exists(dst):
            if a.verify_only:
                print(f"  {n:18} MISSING"); bad += 1; continue
            if not BASE_URL:
                print(f"  {n:18} MISSING - set CARBENCH_WEIGHTS_URL or see checkpoints/README.md")
                bad += 1; continue
            url = BASE_URL.rstrip("/") + "/" + meta["file"]
            print(f"  {n:18} downloading {meta['bytes']/1048576:.1f} MB ...", flush=True)
            os.makedirs(CKPT_DIR, exist_ok=True)
            urllib.request.urlretrieve(url, dst)
        got = sha256(dst)
        ok = got == meta["sha256"]
        bad += (not ok)
        print(f"  {n:18} {'OK  ' if ok else 'SHA MISMATCH'} {meta['bytes']/1048576:7.1f} MB  reproduces {meta['reproduces']}")
    print("\nall checkpoints verified" if not bad else f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
