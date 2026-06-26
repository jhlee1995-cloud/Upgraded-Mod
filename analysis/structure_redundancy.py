"""
analysis/structure_redundancy.py -- are the structure axes 3 or 1?

In mixed streams the 4 single-batch axes showed |corr| 0.90-0.99 (DEVIATION vs the
3 structure axes negative; the 3 structure axes mutually +0.96-0.99). That could be
genuine redundancy OR a clean-corrupt COMMON MODE (everything moves together as a
batch goes clean->corrupt). This script separates the two, in one run:

  [1] WITHIN-GROUP correlation: correlation computed SEPARATELY within clean batches
      and within each corruption's batches. If the +0.99 collapses within-group, it
      was common mode (co-quiet), not redundancy. If it persists, the axes are
      genuinely redundant.
  [2] PER-AXIS unique discrimination: for each disturbance, does each axis separate
      it from clean, and does any axis UNIQUELY separate something the others miss?
  [3] SPLIT test: across disturbances, do the structure axes ever DISAGREE (one high,
      another low)? Synthetic showed confidence_split separated CONSENSUS 0.50 from
      SUBNET_CONSENSUS 0.97 -- does any real disturbance do that?

Needs diag caches (diag_clean, diag_<corruption>, diag_cifar100) from
`extract.py --diag`.

Usage:
  python -m analysis.structure_redundancy --cache /workspace/cache/run6
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache
from sklearn.metrics import roc_auc_score

AXES = ["DEVIATION", "CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS"]
STRUCT = ["CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS"]


def load_diag(cache):
    diag = {}
    for name in cache.list_caches():
        if name.startswith("diag_"):
            diag[name[5:]] = cache._load(name)[0]
    return diag


def corr_table(V, label):
    C = np.corrcoef(V, rowvar=False)
    print(f"\n  {label} (n={len(V)}):")
    print("    " + " ".join(f"{a[:7]:>8s}" for a in AXES))
    for i, a in enumerate(AXES):
        print(f"  {a[:8]:9s}" + " ".join(f"{C[i,j]:8.2f}" for j in range(len(AXES))))
    # report structure-axis mean pairwise corr
    si = [AXES.index(s) for s in STRUCT]
    sc = [C[si[a], si[b]] for a in range(3) for b in range(a+1, 3)]
    print(f"    structure-axes mean pairwise corr: {np.mean(sc):+.2f}")
    return C


def step1_within_group(diag):
    print("=" * 72)
    print("[1] WITHIN-GROUP CORRELATION (common mode vs genuine redundancy)")
    print("=" * 72)
    print("If structure-axis corr stays ~0.99 within a single group -> genuine redundancy.")
    print("If it drops -> the 0.99 in mixed streams was clean-corrupt common mode.")

    if "clean" in diag:
        corr_table(diag["clean"], "WITHIN clean")
    for g in sorted(diag):
        if g == "clean":
            continue
        corr_table(diag[g], f"WITHIN {g}")


def step2_per_axis_discrimination(diag):
    print("\n" + "=" * 72)
    print("[2] PER-AXIS DISCRIMINATION (clean vs each disturbance) + unique coverage")
    print("=" * 72)
    if "clean" not in diag:
        print("  no diag_clean; cannot compute clean-vs-disturbance AUC")
        return
    clean = diag["clean"]
    disturbances = [g for g in sorted(diag) if g != "clean"]

    print(f"\n{'disturbance':22s} " + " ".join(f"{a[:8]:>9s}" for a in AXES))
    print("-" * (24 + 10 * len(AXES)))
    auc_matrix = {}
    for d in disturbances:
        aucs = []
        for i, a in enumerate(AXES):
            sc, sd = clean[:, i], diag[d][:, i]
            y = np.r_[np.zeros(len(sc)), np.ones(len(sd))]
            s = np.r_[sc, sd]
            auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
            # use directional AUC (max of auc, 1-auc) since some axes fire negative
            aucs.append(max(auc, 1 - auc))
        auc_matrix[d] = aucs
        print(f"{d:22s} " + " ".join(f"{a:9.2f}" for a in aucs))

    # unique coverage: for each structure axis, is there a disturbance only IT catches?
    print("\n  unique coverage among structure axes (AUC>0.7 for one, <0.6 for others):")
    si = [AXES.index(s) for s in STRUCT]
    any_unique = False
    for d in disturbances:
        a = auc_matrix[d]
        for k, s in zip(si, STRUCT):
            others = [a[o] for o in si if o != k]
            if a[k] > 0.7 and all(o < 0.6 for o in others):
                print(f"    {s} uniquely catches {d} ({a[k]:.2f} vs others {[round(o,2) for o in others]})")
                any_unique = True
    if not any_unique:
        print("    NONE -- no structure axis uniquely catches any disturbance here.")
        print("    (consistent with redundancy; needs more disturbance types to confirm)")


def step3_split_test(diag):
    print("\n" + "=" * 72)
    print("[3] SPLIT TEST (do structure axes ever DISAGREE on a disturbance?)")
    print("=" * 72)
    print("Synthetic: confidence_split gave CONSENSUS 0.50 vs SUBNET_CONSENSUS 0.97.")
    print("Does any real disturbance split the structure axes' clean-vs-dist AUC?\n")
    if "clean" not in diag:
        print("  no diag_clean")
        return
    clean = diag["clean"]
    si = [AXES.index(s) for s in STRUCT]
    max_split = 0
    for d in sorted(diag):
        if d == "clean":
            continue
        aucs = []
        for k in si:
            sc, sd = clean[:, k], diag[d][:, k]
            y = np.r_[np.zeros(len(sc)), np.ones(len(sd))]
            s = np.r_[sc, sd]
            auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
            aucs.append(max(auc, 1 - auc))
        split = max(aucs) - min(aucs)
        max_split = max(max_split, split)
        flag = " <-- SPLIT" if split > 0.2 else ""
        print(f"  {d:22s} " + " ".join(f"{s[:6]}={a:.2f}" for s, a in zip(STRUCT, aucs))
              + f"  (spread {split:.2f}){flag}")
    print(f"\n  max split across disturbances: {max_split:.2f}")
    if max_split > 0.2:
        print("  -> structure axes DO split on some disturbance -> NOT fully redundant.")
    else:
        print("  -> structure axes never split here -> they track together (redundant on")
        print("     these disturbances; a splitting stimulus like energy-normal type-b")
        print("     (iSUN) is still needed to be sure).")


def main(args):
    cache = FrameCache(args.cache)
    diag = load_diag(cache)
    if not diag:
        print("No diag_* caches found. Re-extract with: extract.py --diag --cifar10c")
        print(f"(caches present: {cache.list_caches()})")
        return
    print(f"diagnostic caches: {list(diag)}\n")
    step1_within_group(diag)
    step2_per_axis_discrimination(diag)
    step3_split_test(diag)
    print("\n" + "=" * 72)
    print("DONE -- structure-axis redundancy analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
