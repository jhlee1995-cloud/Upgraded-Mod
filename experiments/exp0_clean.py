"""
exp0_clean.py -- robustness recheck of the Path-3 gate on CLEAN data, large sample.

The first exp0 run used 80 axis vectors pooled from OOD point caches (cifar100+svhn),
with a worrying bootstrap spread (0.37). Two fixes here:
  1. CLEAN basis: the gate should measure the axis-covariance of the NORMAL aligned
     state (clean), not of OOD batches. We slice cached clean_acts into many small
     batches and compute axis vectors on each -> the clean axis cloud.
  2. LARGE sample: 5120 clean activations / batch_size gives hundreds of vectors,
     not 80 -> the bootstrap spread should shrink if the plateau is real.

If plateau holds with low spread on clean+large-sample -> Path 3 is solid.
If it collapses or spread stays high -> the first plateau was small-sample noise.

Run on pod:
  python -m experiments.exp0_clean --cache /runpod-volume/cache/run1 --batch 32
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache
from extract.axis_registry import AxisRef, SINGLE_BATCH_AXES
from experiments.exp0 import sweep_curve, verdict


def main(args):
    cache = FrameCache(args.cache)
    clean = cache.clean_acts()                      # (5120, 64) real clean activations
    print(f"exp0_clean -- Path-3 gate recheck | cache={args.cache}")
    print(f"clean activations: {clean.shape}")

    # need labels to build the AxisRef (per-subnet class centers). We don't have
    # them cached, so derive pseudo-labels via k-means on clean (10 classes) --
    # the subnet centers only need a reasonable partition, not true labels.
    from sklearn.cluster import KMeans
    pseudo = KMeans(n_clusters=args.n_classes, n_init=10, random_state=0).fit(clean).labels_

    # split clean: half to fit ref, half to slice into axis-vector batches
    n = len(clean)
    half = n // 2
    ref = AxisRef(clean[:half], pseudo[:half], n_classes=args.n_classes)

    # slice the second half into many small batches -> axis vectors
    rest = clean[half:]
    axis_order = list(SINGLE_BATCH_AXES)
    n_batches = len(rest) // args.batch
    vectors = []
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rest))
    for b in range(n_batches):
        idx = perm[b * args.batch:(b + 1) * args.batch]
        X = rest[idx]
        vectors.append([SINGLE_BATCH_AXES[a](X, ref).mean() for a in axis_order])
    V = np.array(vectors)
    print(f"clean axis vectors: {V.shape}  (vs 80 in first run)")

    taus = np.linspace(0, 0.95, 20)
    curve, per_boot, max_off = sweep_curve(V, taus)
    v = verdict(taus, curve)
    print(f"\nmax |off-diagonal corr| = {max_off:.2f}")
    print("sign-stable pairs vs tau: " + " ".join(f"{c:2d}" for c in curve[::3])
          + f"  (tau={', '.join(f'{t:.2f}' for t in taus[::3])})")
    print(f"per-bootstrap spread (seed stability): {per_boot.std(0).mean():.2f}  "
          f"(first run: 0.37; lower = more stable)")
    print(f"\nVERDICT: {v}")

    # also show the raw correlation matrix so we SEE which axes are coupled
    C = np.corrcoef(V, rowvar=False)
    print("\naxis correlation matrix (clean):")
    print("            " + " ".join(f"{a[:8]:>9s}" for a in axis_order))
    for i, a in enumerate(axis_order):
        print(f"{a:12s}" + " ".join(f"{C[i,j]:9.2f}" for j in range(len(axis_order))))
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--batch", type=int, default=32, help="batch size for axis vectors")
    ap.add_argument("--n-classes", type=int, default=10)
    main(ap.parse_args())
