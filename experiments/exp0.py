"""
experiments/exp0.py -- covariance off-diagonal sweep (the Path-3 gate).

Validated in toy_redesign.py: sweeping the off-diagonal |correlation| threshold
and reading the CURVE SHAPE separates curved / weak / diagonal axis structure,
threshold-independently. On REAL cached axis vectors this decides whether the
alignment-topology layer is meaningful:

  plateau (curved)   -> axis space is curved; topology meaningful; Path 3 lives
  gentle (weak)      -> weak structure; possible but weak foundation
  collapse (diagonal)-> axes independent; 64-cell occupancy near-best; rethink topology

Run:
  python -m experiments.exp0 --cache /runpod-volume/cache/run1
"""
import argparse
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache


def offdiag_abs(C):
    iu = np.triu_indices(C.shape[0], k=1)
    return np.abs(C[iu])


def sweep_curve(vectors, taus, n_boot=50):
    """Bootstrap the off-diagonal correlation; for each tau count sign-stable
    pairs above it. Returns mean curve + per-bootstrap curves (seed-stability)."""
    D = vectors.shape[1]
    iu = np.triu_indices(D, k=1)
    n_pairs = len(iu[0])
    boot_corrs = np.zeros((n_boot, n_pairs))
    n = len(vectors)
    for b in range(n_boot):
        idx = np.random.default_rng(b).integers(0, n, n)
        sample = vectors[idx]
        # guard: drop the bootstrap if any axis is constant (std=0 -> nan corr)
        stds = sample.std(0)
        if np.any(stds < 1e-12):
            # add tiny jitter to constant axes so corrcoef is defined (->~0 corr)
            sample = sample + np.random.default_rng(b + 9999).normal(0, 1e-9, sample.shape)
        C = np.corrcoef(sample, rowvar=False)
        C = np.nan_to_num(C, nan=0.0)
        boot_corrs[b] = C[iu]
    sign_frac = np.mean(np.sign(boot_corrs) == np.sign(np.median(boot_corrs, 0)), 0)
    sign_stable = sign_frac >= 0.90
    median_abs = np.abs(np.median(boot_corrs, 0))
    mean_curve = np.array([np.sum((median_abs >= t) & sign_stable) for t in taus])
    per_boot = np.array([[np.sum(np.abs(boot_corrs[b]) >= t) for t in taus]
                         for b in range(n_boot)])
    return mean_curve, per_boot, np.max(median_abs)


def verdict(taus, curve):
    c = curve.astype(float)
    if c[0] == 0:
        return "collapse (near-diagonal)"
    half = c[0] / 2.0
    idx = np.argmax(c <= half) if np.any(c <= half) else len(c) - 1
    tau_half = taus[idx]
    if tau_half >= 0.45:
        return "plateau (curved -> topology meaningful, Path 3 lives)"
    elif tau_half >= 0.15:
        return "gentle slope (weak structure)"
    return "collapse (near-diagonal -> axes independent, rethink topology)"


def main(args):
    cache = FrameCache(args.cache)
    print(f"exp0 -- covariance off-diagonal gate | cache={args.cache} "
          f"| synthetic={cache.is_synthetic()}")
    # use clean activations projected to axis space? No -- use the POINT vectors
    # (per-batch axis vectors), which live in axis space directly. Pool clean-ish
    # points: here we use the clean_acts reduced to axis vectors is not stored;
    # instead exp0 runs on the disturbance point caches' clean baseline. For the
    # real run, point_<dataset> rows ARE axis vectors -> stack available ones.
    vecs = []
    for name in cache.list_caches():
        if name.startswith("point_"):
            arr, _ = cache._load(name)
            vecs.append(arr)
    if not vecs:
        raise RuntimeError("no point_* caches found; run extract.py first")
    V = np.vstack(vecs)
    print(f"axis vectors: {V.shape} (pooled from point caches)")

    taus = np.linspace(0, 0.95, 20)
    curve, per_boot, max_off = sweep_curve(V, taus)
    v = verdict(taus, curve)
    print(f"\nmax |off-diagonal corr| = {max_off:.2f}")
    print(f"sign-stable pairs vs tau: " + " ".join(f"{c:2d}" for c in curve[::3])
          + f"  (tau={', '.join(f'{t:.2f}' for t in taus[::3])})")
    print(f"per-bootstrap spread (seed stability): {per_boot.std(0).mean():.2f}")
    print(f"\nVERDICT: {v}")
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
