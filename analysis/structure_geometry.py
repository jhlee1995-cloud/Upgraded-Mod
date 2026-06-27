"""
analysis/structure_geometry.py -- does penult space have VALLEYS, and where is type-b?

The user's question: is the activation space actually shaped like valleys (clusters with
gaps between them) so that 'how stuck between clusters' is a meaningful measure? And what
does CLUSTER_DISTANCE actually capture? This probes the raw geometry before trusting any
distance-based axis.

Runs, in one go:
  [1] SEPARATION: between/within ratio + per-class spread. Do valleys exist at all?
  [2] SHAPE: per-class covariance eigenvalue anisotropy. Spherical (mean_nearest ok) or
      elongated (mahalanobis needed)? -- answers 'do we need a different distance formula'.
  [3] TYPE-B POSITION: where do misclassified samples sit? On a ridge between clusters
      (margin small) or deep in the WRONG cluster (margin large)? + their confidence.
  [4] VALLEY MEASURES vs standard axes: do margin/entropy separate correct-vs-type-b where
      CLUSTER_DISTANCE saturates? (uses diag_cifar10_correct / _confwrong if present)

Needs <cache>/structure/cluster_geometry.npz (from extract --cluster-struct) and
optionally diag_cifar10_correct / diag_cifar10_confwrong (from --typeb) which now carry
VALLEY_MARGIN, VALLEY_ENTROPY columns.

Usage:
  python -m analysis.structure_geometry --cache /workspace/cache/run10
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache
from sklearn.metrics import roc_auc_score

# 6-col diag format: 4 standard + 2 valley
DIAG_COLS = ["DEVIATION", "CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS",
             "VALLEY_MARGIN", "VALLEY_ENTROPY"]


def dir_auc(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    s = np.r_[a, b]
    if len(np.unique(s)) < 2:
        return 0.5
    return max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))


def step1_separation(G):
    print("=" * 72)
    print("[1] SEPARATION -- do valleys exist? (between-cluster vs within-cluster)")
    print("=" * 72)
    within, between = G["within"], G["between"]
    ev = G["explained_var"]
    print(f"\n  within-class spread:  mean {within.mean():.3f}  range [{within.min():.3f}, {within.max():.3f}]")
    print(f"  between-class dist:   mean {np.nanmean(between):.3f}  min {np.nanmin(between):.3f}")
    sep = np.nanmean(between) / (within.mean() + 1e-9)
    min_sep = np.nanmin(between) / (within.max() + 1e-9)
    print(f"  separation ratio (mean between / mean within): {sep:.2f}")
    print(f"  worst-case ratio (min between / max within):    {min_sep:.2f}")
    if sep > 2.0:
        print("  >>> STRONG valleys: clusters well separated -> distance axes well-founded.")
    elif sep > 1.3:
        print("  >>> MODERATE valleys: separated but with overlap -> distance works but")
        print("      valley measures (margin/entropy) may add resolution.")
    else:
        print("  >>> WEAK/NO valleys: clusters overlap heavily -> 'distance to center' is")
        print("      shaky; mean_nearest may be capturing mostly energy/spread, not cluster id.")
    print(f"\n  PCA explained variance (top-10): {ev.round(3)}")
    print(f"  -> top-2 explain {ev[:2].sum()*100:.0f}%; if low, 2D plots will look mushed")
    print("     even if real (high-dim) valleys exist.")


def step2_shape(G):
    print("\n" + "=" * 72)
    print("[2] SHAPE -- spherical or elongated clusters? (mean_nearest vs mahalanobis)")
    print("=" * 72)
    eigs = G["eigs"]  # (nc, 10)
    # anisotropy per class = top eigenvalue / mean of the rest
    aniso = []
    for c in range(len(eigs)):
        e = eigs[c][eigs[c] > 0]
        if len(e) > 1:
            aniso.append(e[0] / (e[1:].mean() + 1e-9))
    aniso = np.array(aniso)
    print(f"\n  per-class anisotropy (top eig / rest): mean {aniso.mean():.1f}, "
          f"range [{aniso.min():.1f}, {aniso.max():.1f}]")
    print(f"  eigenvalue decay (class 0): {eigs[0][eigs[0]>0][:5].round(2)}")
    if aniso.mean() > 5:
        print("  >>> ELONGATED clusters: covariance is anisotropic -> Euclidean mean_nearest")
        print("      distorts; MAHALANOBIS distance would respect cluster shape. A distance")
        print("      formula variant is justified.")
    elif aniso.mean() > 2.5:
        print("  >>> MILDLY elongated: Euclidean is ok but mahalanobis might sharpen.")
    else:
        print("  >>> ~SPHERICAL: mean_nearest (Euclidean) is well-matched; no need for")
        print("      mahalanobis on shape grounds.")


def step3_typeb_position(G):
    print("\n" + "=" * 72)
    print("[3] TYPE-B POSITION -- where do misclassified samples sit in the valleys?")
    print("=" * 72)
    centers = G["centers"]
    proj = G["proj2d"]
    lab, pred, conf = G["labels"], G["preds"], G["conf"]
    wrong = pred != lab
    print(f"\n  {wrong.sum()} misclassified of {len(lab)} ({100*wrong.mean():.1f}%)")
    if wrong.sum() == 0:
        print("  no misclassified samples collected")
        return

    # for each sample, compute distance to its TRUE center and its PREDICTED center
    # (in full space, using the per-class centers)
    def full_dists(idx):
        x = proj  # use 2D proj as a stand-in only for ridge geometry; full uses centers
        return None
    # better: recompute in full space requires raw feats (not saved); use centers + the
    # fact that we have labels/preds. Approx via 2D proj distances to projected centers.
    cproj = np.zeros((len(centers), 2))
    cm = proj.mean(0)
    # project class centers the same way: centers are in full space; we don't have Vt here,
    # so approximate each class's projected center as the mean of its samples' projections.
    for c in range(len(centers)):
        m = lab == c
        if m.any():
            cproj[c] = proj[m].mean(0)

    # margin in projection: dist to nearest projected center minus to second nearest
    d_all = np.linalg.norm(proj[:, None] - cproj[None], axis=2)  # (N, nc)
    ds = np.sort(d_all, axis=1)
    margin = ds[:, 1] - ds[:, 0]

    print(f"  margin (2nd-nearest minus nearest center, in 2D proj):")
    print(f"    correct samples:      mean {margin[~wrong].mean():.3f}")
    print(f"    misclassified (type-b): mean {margin[wrong].mean():.3f}")
    print(f"    confident-wrong (>0.7): mean {margin[wrong & (conf>0.7)].mean():.3f}"
          if (wrong & (conf > 0.7)).any() else "    (no confident-wrong)")
    if margin[wrong].mean() < margin[~wrong].mean() * 0.7:
        print("  >>> type-b sits on RIDGES (small margin): between clusters, not deep in one.")
        print("      -> 'valley between clusters' measure (small margin) flags type-b.")
    else:
        print("  >>> type-b sits DEEP in (wrong) clusters (margin like correct): it commits")
        print("      to a wrong valley -> margin does NOT flag it; distance-to-TRUE-center")
        print("      would, but that needs the label. This is why type-b is hard.")
    print(f"\n  confidence of misclassified: mean {conf[wrong].mean():.3f} "
          f"(high = confident-wrong = the dangerous type-b)")


def step4_valley_measures(cache):
    print("\n" + "=" * 72)
    print("[4] VALLEY MEASURES vs STANDARD AXES on correct-vs-type-b")
    print("=" * 72)
    diag = {}
    for name in cache.list_caches():
        if name.startswith("diag_"):
            diag[name[5:]] = cache._load(name)[0]
    ref = diag.get("cifar10_correct")
    if ref is None or "cifar10_confwrong" not in diag:
        print("  need diag_cifar10_correct and diag_cifar10_confwrong (run --typeb)")
        return
    if ref.shape[1] < 6:
        print(f"  diag caches have {ref.shape[1]} cols (no valley geometry); re-extract.")
        return
    tb = diag["cifar10_confwrong"]
    print(f"\n  correct-vs-confwrong AUC per measure:")
    print(f"  {'measure':18s} {'AUC':>8s}")
    for i, name in enumerate(DIAG_COLS):
        auc = dir_auc(ref[:, i], tb[:, i])
        star = "  <-- valley" if name.startswith("VALLEY") else ""
        print(f"  {name:18s} {auc:8.2f}{star}")
    # which separates type-b best?
    aucs = {name: dir_auc(ref[:, i], tb[:, i]) for i, name in enumerate(DIAG_COLS)}
    best = max(aucs, key=aucs.get)
    print(f"\n  best type-b separator: {best} ({aucs[best]:.2f})")
    vm = aucs["VALLEY_MARGIN"]; ve = aucs["VALLEY_ENTROPY"]; cd = aucs["CLUSTER_DISTANCE"]
    if max(vm, ve) > cd + 0.1:
        print(f"  >>> a VALLEY measure beats CLUSTER_DISTANCE on type-b ({max(vm,ve):.2f} vs"
              f" {cd:.2f}) -> the valley formulation captures type-b better. Consider it.")
    elif cd > max(vm, ve) + 0.1:
        print(f"  >>> CLUSTER_DISTANCE still best ({cd:.2f}) -> valley measures don't help here.")
    else:
        print(f"  >>> comparable -> no clear winner on this type-b set.")


def main(args):
    cache = FrameCache(args.cache)
    gpath = os.path.join(args.cache, "structure", "cluster_geometry.npz")
    if not os.path.exists(gpath):
        print(f"No cluster_geometry.npz at {gpath}")
        print("Run: extract --cluster-struct --typeb")
        return
    G = np.load(gpath)
    step1_separation(G)
    step2_shape(G)
    step3_typeb_position(G)
    step4_valley_measures(cache)
    print("\n" + "=" * 72)
    print("DONE -- valley/cluster-geometry analysis complete.")
    print("  proj2d + labels are in cluster_geometry.npz for plotting if you want to SEE")
    print("  the valleys (scatter proj2d colored by label, overlay misclassified).")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
