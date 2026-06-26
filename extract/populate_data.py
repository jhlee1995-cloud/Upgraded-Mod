"""
populate_data.py -- one-time dataset population on the network volume.

Run this ONCE on any pod with the volume mounted. It downloads every dataset the
project needs into a STANDARD layout under <volume>/datasets/, and writes a
manifest recording canonical paths, file counts, and checksums.

Why: the recurring pain was a mount-path mismatch (populate pod saw /workspace,
later pod saw /runpod-volume). The fix:
  - everything goes under one root you pass as --volume (e.g. /runpod-volume)
  - the manifest records paths RELATIVE to that root
  - later code reads the manifest and joins with its OWN --volume, so the
    absolute mount point can differ between pods without breaking anything.

Usage on pod:
  python populate_data.py --volume /runpod-volume          # download everything
  python populate_data.py --volume /runpod-volume --verify  # just re-check manifest

Datasets:
  cifar10   (torchvision)         -- clean coordinate frame
  cifar100  (torchvision)         -- near-OOD (type-b bearing)
  svhn      (torchvision)         -- far-OOD control
  cifar10c  (manual URL / skip)   -- corruption benchmark (large; see note)
  isun      (manual / skip)       -- near-OOD for distance-vs-deviation (manual add)
"""
import argparse
import hashlib
import json
import os
import time


DATASETS_SUBDIR = "datasets"
MANIFEST_NAME = "manifest.json"


def sha256_of_dir(path, max_files=50):
    """Cheap integrity fingerprint: hash of sorted (relpath, size) for up to
    max_files files. Not a full content hash (datasets are large) but catches
    missing/extra/truncated files."""
    h = hashlib.sha256()
    entries = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, path)
            try:
                entries.append((rel, os.path.getsize(fp)))
            except OSError:
                pass
    entries.sort()
    for rel, size in entries[:max_files]:
        h.update(rel.encode())
        h.update(str(size).encode())
    return h.hexdigest()[:16], len(entries)


def count_dir(path):
    n = 0
    for _, _, files in os.walk(path):
        n += len(files)
    return n


def populate(args):
    root = os.path.join(args.volume, DATASETS_SUBDIR)
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, MANIFEST_NAME)

    if args.verify:
        return verify(args, root, manifest_path)

    import torchvision

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "volume_at_creation": args.volume,
        "note": "paths below are RELATIVE to <volume>/datasets; join with your own --volume",
        "datasets": {},
    }

    # --- torchvision datasets (auto-download) ---
    tv_specs = [
        ("cifar10",  lambda d: torchvision.datasets.CIFAR10(d, train=False, download=True)),
        ("cifar10_train", lambda d: torchvision.datasets.CIFAR10(d, train=True, download=True)),
        ("cifar100", lambda d: torchvision.datasets.CIFAR100(d, train=False, download=True)),
        ("svhn",     lambda d: torchvision.datasets.SVHN(d, split="test", download=True)),
    ]
    for name, loader in tv_specs:
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        ddir = os.path.join(root, name)
        os.makedirs(ddir, exist_ok=True)
        print(f"[download] {name} -> {ddir}")
        try:
            loader(ddir)
            fp, nfiles = sha256_of_dir(ddir)
            manifest["datasets"][name] = {
                "rel_path": os.path.relpath(ddir, args.volume),
                "n_files": nfiles,
                "fingerprint": fp,
                "source": "torchvision",
            }
            print(f"   ok: {nfiles} files, fp={fp}")
        except Exception as e:
            print(f"   FAILED: {repr(e)[:120]}")
            manifest["datasets"][name] = {"error": repr(e)[:200]}

    # --- manual datasets (CIFAR-10-C, iSUN): record placeholder, instruct ---
    for name, hint in [
        ("cifar10c", "CIFAR-10-C: download CIFAR-10-C.tar from zenodo, extract .npy files here"),
        ("isun", "iSUN: add via dataset upload, place images here"),
    ]:
        ddir = os.path.join(root, name)
        if os.path.isdir(ddir) and count_dir(ddir) > 0:
            fp, nfiles = sha256_of_dir(ddir)
            manifest["datasets"][name] = {
                "rel_path": os.path.relpath(ddir, args.volume),
                "n_files": nfiles, "fingerprint": fp, "source": "manual",
            }
            print(f"[manual] {name}: found {nfiles} files, fp={fp}")
        else:
            manifest["datasets"][name] = {
                "rel_path": os.path.relpath(ddir, args.volume),
                "n_files": 0, "status": "NOT POPULATED", "hint": hint,
            }
            print(f"[manual] {name}: not present -- {hint}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest written: {manifest_path}")
    print("done. later pods: read this manifest, join rel_path with your own --volume.")
    return manifest


def verify(args, root, manifest_path):
    if not os.path.exists(manifest_path):
        print(f"no manifest at {manifest_path} -- run populate first")
        return None
    manifest = json.load(open(manifest_path))
    print(f"verifying manifest (created {manifest.get('created')})")
    print(f"  volume at creation: {manifest.get('volume_at_creation')}")
    print(f"  volume now:         {args.volume}")
    all_ok = True
    for name, info in manifest["datasets"].items():
        if "error" in info or info.get("n_files", 0) == 0:
            print(f"  [{name:14s}] MISSING / {info.get('status', info.get('error',''))[:40]}")
            all_ok = False
            continue
        ddir = os.path.join(args.volume, info["rel_path"])
        if not os.path.isdir(ddir):
            print(f"  [{name:14s}] PATH MISSING: {ddir}")
            all_ok = False
            continue
        fp, nfiles = sha256_of_dir(ddir)
        match = (fp == info["fingerprint"])
        print(f"  [{name:14s}] {nfiles} files, fp={fp} "
              f"{'OK' if match else 'FINGERPRINT MISMATCH'}")
        all_ok = all_ok and match
    print(f"\noverall: {'ALL OK' if all_ok else 'ISSUES FOUND'}")
    return manifest


def resolve_path(volume, dataset_name):
    """Helper for later code: get the absolute path of a dataset on THIS pod's
    volume, via the manifest. Import this in extract/layer_sweep."""
    manifest_path = os.path.join(volume, DATASETS_SUBDIR, MANIFEST_NAME)
    manifest = json.load(open(manifest_path))
    info = manifest["datasets"].get(dataset_name)
    if info is None or "rel_path" not in info:
        raise KeyError(f"{dataset_name} not in manifest")
    return os.path.join(volume, info["rel_path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True,
                    help="volume mount root on THIS pod (e.g. /runpod-volume)")
    ap.add_argument("--verify", action="store_true", help="verify existing manifest only")
    ap.add_argument("--skip", nargs="*", default=[], help="dataset names to skip")
    args = ap.parse_args()
    populate(args)
