"""
analysis/typeb_analysis.py -- the decisive type-b test using iSUN.

iSUN is energy-normal near-OOD. It is the SPLITTING STIMULUS missing from the
redundancy analysis: on type-a-like errors (corruption, CIFAR-100) all 3 structure
axes co-fire, so they look redundant. type-b should make CLUSTER_DISTANCE fire while
CONSENSUS stays low -> if the structure axes SPLIT here, they are genuinely 3 (keep
CLD as type-b's unique detector); if they don't, they're redundant (7->5 possible).

Also resolves CLUSTER_DISTANCE vs DEVIATION orthogonality (memory#18's last false-
redundancy pair): synthetic type-b was always energy-large so both fired; real
energy-normal iSUN is the input that separates them.

Tests, in one run:
  [1] ENERGY PROFILE: is iSUN energy-normal (DEVIATION near clean) unlike corruption?
  [2] STRUCTURE-AXIS SPLIT on iSUN: does CLD fire (high AUC) while CONSENSUS stays low?
  [3] CLD vs DEV orthogonality: on iSUN, is CLD strong while DEV weak? (the separating case)
  [4] map verdict: structure axes 3-or-1, and CLD load-bearing or not.

Needs diag_clean, diag_isun (+ diag_<corruption> for contrast) from extract --diag --isun.

Usage:
  python -m analysis.typeb_analysis --cache /workspace/cache/run8
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache
from sklearn.metrics import roc_auc_score

AXES = ["DEVIATION", "CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS"]


def load_diag(cache):
    diag = {}
    for name in cache.list_caches():
        if name.startswith("diag_"):
            diag[name[5:]] = cache._load(name)[0]
    return diag


def dir_auc(clean_col, dist_col):
    """directional AUC: max(auc, 1-auc) since axes may fire in either direction."""
    y = np.r_[np.zeros(len(clean_col)), np.ones(len(dist_col))]
    s = np.r_[clean_col, dist_col]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def step1_energy_profile(diag):
    print("=" * 72)
    print("[1] ENERGY PROFILE -- is iSUN energy-normal (unlike corruption)?")
    print("=" * 72)
    if "clean" not in diag:
        print("  no diag_clean")
        return
    di = AXES.index("DEVIATION")
    clean_dev = diag["clean"][:, di].mean()
    print(f"\n  clean DEVIATION mean: {clean_dev:.3f}")
    print(f"  {'dataset':18s} {'DEV mean':>10s} {'vs clean':>10s}")
    for d in sorted(diag):
        if d == "clean":
            continue
        dev = diag[d][:, di].mean()
        print(f"  {d:18s} {dev:10.3f} {dev - clean_dev:+10.3f}")
    print("\n  -> corruption/near-OOD LOWER deviation (energy decrease). If iSUN is")
    print("     CLOSER to clean than corruption is, it's the energy-normal case that")
    print("     separates CLUSTER_DISTANCE from DEVIATION.")


def step2_structure_split(diag):
    print("\n" + "=" * 72)
    print("[2] STRUCTURE-AXIS SPLIT -- does iSUN split CLD from CONSENSUS? (type-b sig)")
    print("=" * 72)
    print("type-b signature: CLUSTER_DISTANCE fires (high AUC) but CONSENSUS stays low.")
    print("If structure axes SPLIT on iSUN -> genuinely 3 axes (keep CLD). If not -> redundant.\n")
    if "clean" not in diag:
        print("  no diag_clean")
        return
    clean = diag["clean"]
    print(f"  {'dataset':18s} " + " ".join(f"{a[:8]:>9s}" for a in AXES) + "   spread(struct)")
    print("  " + "-" * 78)
    for d in sorted(diag):
        if d == "clean":
            continue
        aucs = [dir_auc(clean[:, i], diag[d][:, i]) for i in range(len(AXES))]
        struct = [aucs[AXES.index(s)] for s in ["CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS"]]
        spread = max(struct) - min(struct)
        tag = ""
        if d == "isun":
            tag = " <== type-b candidate"
        print(f"  {d:18s} " + " ".join(f"{a:9.2f}" for a in aucs) + f"   {spread:.2f}{tag}")

    if "isun" in diag:
        a_isun = [dir_auc(clean[:, i], diag["isun"][:, i]) for i in range(len(AXES))]
        cld = a_isun[AXES.index("CLUSTER_DISTANCE")]
        con = a_isun[AXES.index("CONSENSUS")]
        print(f"\n  iSUN: CLUSTER_DISTANCE={cld:.2f}, CONSENSUS={con:.2f}")
        if cld > 0.7 and con < cld - 0.15:
            print("  >>> SPLIT CONFIRMED: CLD fires, CONSENSUS lags -> CLD is type-b's unique")
            print("      detector -> KEEP all 3 structure axes (do NOT collapse to 1).")
        elif cld > 0.7 and con > 0.7:
            print("  >>> NO SPLIT: both fire on iSUN -> structure axes still redundant here.")
            print("      (iSUN may be more type-a than type-b; or genuinely 7->5 possible.)")
        else:
            print(f"  >>> WEAK: neither strongly fires (CLD {cld:.2f}) -> iSUN may be far-OOD or")
            print("      the axes miss this; inspect energy profile + raw distances.")


def step3_cld_dev_orthogonality(diag):
    print("\n" + "=" * 72)
    print("[3] CLUSTER_DISTANCE vs DEVIATION orthogonality (last false-redundancy pair)")
    print("=" * 72)
    print("Synthetic type-b was energy-LARGE so DEV and CLD both fired (looked redundant).")
    print("Energy-normal iSUN should make CLD fire while DEV stays weak -> they separate.\n")
    if "clean" not in diag or "isun" not in diag:
        print("  need diag_clean and diag_isun")
        return
    clean = diag["clean"]
    di, ci = AXES.index("DEVIATION"), AXES.index("CLUSTER_DISTANCE")
    print(f"  {'dataset':18s} {'DEV auc':>9s} {'CLD auc':>9s} {'CLD-DEV':>9s}")
    for d in sorted(diag):
        if d == "clean":
            continue
        dev = dir_auc(clean[:, di], diag[d][:, di])
        cld = dir_auc(clean[:, ci], diag[d][:, ci])
        print(f"  {d:18s} {dev:9.2f} {cld:9.2f} {cld - dev:+9.2f}")
    dev_i = dir_auc(clean[:, di], diag["isun"][:, di])
    cld_i = dir_auc(clean[:, ci], diag["isun"][:, ci])
    print(f"\n  iSUN: DEV={dev_i:.2f}, CLD={cld_i:.2f}")
    if cld_i > dev_i + 0.15:
        print("  >>> ORTHOGONAL: on energy-normal iSUN, CLD fires but DEV doesn't ->")
        print("      CLUSTER_DISTANCE is NOT redundant with DEVIATION -> KEEP both.")
    elif abs(cld_i - dev_i) < 0.1 and cld_i > 0.7:
        print("  >>> STILL CO-FIRING: both fire on iSUN too -> iSUN isn't energy-normal")
        print("      enough, or they really are redundant. Check energy profile [1].")
    else:
        print("  >>> inconclusive; inspect energy profile.")


def main(args):
    cache = FrameCache(args.cache)
    diag = load_diag(cache)
    if "isun" not in diag:
        print("No diag_isun cache. First: download_isun.py, then extract --diag --isun.")
        print(f"(diag caches present: {list(diag)})")
        return
    print(f"diagnostic caches: {list(diag)}\n")
    step1_energy_profile(diag)
    step2_structure_split(diag)
    step3_cld_dev_orthogonality(diag)
    print("\n" + "=" * 72)
    print("DONE -- type-b / iSUN analysis complete.")
    print("Map updates: row 4 (type-b) cells; structure-axis column count decision.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
