"""
cache_audit.py -- auditable cache layer.

Every cache (6/7-axis .npy vectors) is written WITH a sidecar metadata file so its
provenance is always traceable: was it synthetic or real, which backbone/layer,
which dataset (fingerprint), which axis formulas, when. This is the core of the
new repo's bug-management discipline -- no more "is this cache stale / synthetic /
from which run?" guesswork.

Layout:
  cache/
    <name>.npy            # the (N, n_axes) vectors
    <name>.meta.json      # provenance sidecar
    AUDIT.json            # rolling index of all caches (rebuilt by `audit`)

Use:
  from cache_audit import save_cache, load_cache
  save_cache(arr, "clean", meta=dict(source="real", backbone=..., layer=..., ...))
  arr, meta = load_cache("clean")

  python cache_audit.py --dir cache --audit     # print provenance table
  python cache_audit.py --dir cache --verify    # check contract + flag issues
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np


META_SUFFIX = ".meta.json"
AUDIT_NAME = "AUDIT.json"
AXIS_NAMES = ["DEVIATION", "CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS",
              "DRIFT_COH", "PERSIST", "CLUST_DRIFT"]


def _arr_fingerprint(arr):
    """content fingerprint of the array (shape + sampled bytes)."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    # sample to keep it cheap on big arrays
    flat = arr.ravel()
    step = max(1, len(flat) // 10000)
    h.update(flat[::step].tobytes())
    return h.hexdigest()[:16]


def save_cache(arr, name, cache_dir, meta=None):
    """Save (N, n_axes) array + provenance sidecar. meta should include at least
    source ('real'|'synthetic'), and for real: backbone, layer, dataset, dataset_fp."""
    os.makedirs(cache_dir, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    npy_path = os.path.join(cache_dir, f"{name}.npy")
    np.save(npy_path, arr)

    full_meta = {
        "name": name,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shape": list(arr.shape),
        "n_axes": int(arr.shape[1]) if arr.ndim == 2 else None,
        "fingerprint": _arr_fingerprint(arr),
        "axis_names": AXIS_NAMES[:arr.shape[1]] if arr.ndim == 2 else None,
        # caller-supplied provenance
        "source": (meta or {}).get("source", "UNKNOWN"),
        "synthetic": (meta or {}).get("source") == "synthetic",
        "backbone": (meta or {}).get("backbone"),
        "layer": (meta or {}).get("layer"),
        "dataset": (meta or {}).get("dataset"),
        "dataset_fingerprint": (meta or {}).get("dataset_fp"),
        "axis_formulas": (meta or {}).get("axis_formulas"),
        "notes": (meta or {}).get("notes"),
    }
    with open(os.path.join(cache_dir, name + META_SUFFIX), "w") as f:
        json.dump(full_meta, f, indent=2)
    return npy_path


def load_cache(name, cache_dir):
    """Load array + its metadata. Raises if metadata missing (enforces provenance)."""
    npy_path = os.path.join(cache_dir, f"{name}.npy")
    meta_path = os.path.join(cache_dir, name + META_SUFFIX)
    arr = np.load(npy_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{name}: missing {META_SUFFIX} -- cache has no provenance, refusing to trust it. "
            f"Re-save via save_cache.")
    meta = json.load(open(meta_path))
    # integrity check
    fp = _arr_fingerprint(arr)
    if fp != meta.get("fingerprint"):
        meta["_WARNING"] = f"fingerprint mismatch (file {fp} != meta {meta.get('fingerprint')})"
    return arr, meta


def audit(cache_dir):
    """Print a provenance table of all caches; rebuild AUDIT.json index."""
    if not os.path.isdir(cache_dir):
        print(f"no cache dir: {cache_dir}")
        return
    names = sorted(f[:-4] for f in os.listdir(cache_dir) if f.endswith(".npy"))
    print("=" * 92)
    print(f"CACHE AUDIT  ({cache_dir})")
    print("=" * 92)
    print(f"{'name':16s} {'src':10s} {'shape':>12s} {'axes':>5s} "
          f"{'backbone':14s} {'layer':8s} {'saved':17s}")
    print("-" * 92)
    index = {}
    n_synth = n_real = n_orphan = 0
    for name in names:
        meta_path = os.path.join(cache_dir, name + META_SUFFIX)
        if not os.path.exists(meta_path):
            print(f"{name:16s} {'NO META':10s} {'?':>12s}  -- ORPHAN (no provenance)")
            n_orphan += 1
            continue
        m = json.load(open(meta_path))
        src = m.get("source", "?")
        if m.get("synthetic"):
            n_synth += 1
        elif src == "real":
            n_real += 1
        shape = "x".join(str(s) for s in m.get("shape", []))
        print(f"{name:16s} {src:10s} {shape:>12s} {str(m.get('n_axes','?')):>5s} "
              f"{str(m.get('backbone',''))[:14]:14s} {str(m.get('layer',''))[:8]:8s} "
              f"{m.get('saved','')[:17]:17s}")
        index[name] = m
    print("-" * 92)
    print(f"total: {len(names)}  |  real: {n_real}  synthetic: {n_synth}  orphan: {n_orphan}")
    if n_synth and n_real:
        print("  ⚠ mixed real + synthetic caches in same dir -- ensure downstream uses the right set")
    if n_orphan:
        print("  ⚠ orphan caches without provenance -- re-save via save_cache or delete")
    with open(os.path.join(cache_dir, AUDIT_NAME), "w") as f:
        json.dump({"audited": time.strftime("%Y-%m-%d %H:%M:%S"), "caches": index}, f, indent=2)
    return index


def verify(cache_dir):
    """Check the Stage-B contract: every cache is (N, n_axes), n_axes consistent,
    fingerprints match, and provenance is present."""
    names = sorted(f[:-4] for f in os.listdir(cache_dir) if f.endswith(".npy"))
    print(f"verifying {len(names)} caches in {cache_dir}")
    n_axes_seen = set()
    ok = True
    for name in names:
        try:
            arr, meta = load_cache(name, cache_dir)
        except FileNotFoundError as e:
            print(f"  [{name:16s}] FAIL: {str(e)[:60]}")
            ok = False
            continue
        if arr.ndim != 2:
            print(f"  [{name:16s}] FAIL: not 2D (shape {arr.shape})")
            ok = False
            continue
        n_axes_seen.add(arr.shape[1])
        warn = meta.get("_WARNING", "")
        status = "OK" if not warn else f"WARN: {warn[:40]}"
        print(f"  [{name:16s}] shape {arr.shape}, src={meta.get('source')}  {status}")
        if warn:
            ok = False
    if len(n_axes_seen) > 1:
        print(f"  ⚠ inconsistent axis count across caches: {n_axes_seen}")
        ok = False
    print(f"\ncontract: {'OK' if ok else 'ISSUES'} (n_axes seen: {n_axes_seen})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cache", help="cache directory")
    ap.add_argument("--audit", action="store_true", help="print provenance table")
    ap.add_argument("--verify", action="store_true", help="verify Stage-B contract")
    ap.add_argument("--demo", action="store_true", help="write a couple of demo caches")
    args = ap.parse_args()

    if args.demo:
        # demonstrate the metadata layout
        save_cache(np.random.randn(500, 7), "clean", args.dir,
                   meta=dict(source="synthetic", backbone="none",
                             layer="penult", dataset="synthetic",
                             axis_formulas={"DEVIATION": "l2_signed"}))
        save_cache(np.random.randn(300, 7), "cifar100", args.dir,
                   meta=dict(source="real", backbone="cifar10_resnet20",
                             layer="penult", dataset="cifar100", dataset_fp="abc123"))
        print("wrote demo caches\n")
    if args.audit:
        audit(args.dir)
    if args.verify:
        verify(args.dir)
    if not (args.audit or args.verify or args.demo):
        ap.print_help()
