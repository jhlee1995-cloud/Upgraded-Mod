"""
analysis/typeb_real.py -- true type-b (CIFAR-10 misclassified) + severity sweep.

iSUN failed as type-b (it's far-OOD, energy drops). The TRUE type-b is CIFAR-10 test
images the model gets WRONG: same distribution (energy-normal) + confident-wrong. This
script tests whether real type-b makes the structure axes SPLIT (CLUSTER_DISTANCE fires,
CONSENSUS silent) -- the decision for whether structure axes are genuinely 3 or 1.

Also runs the SEVERITY SWEEP to escape the AUC ceiling (severity-3 saturated everything
at 1.00; severity-1 should show sub-ceiling AUC where splits become visible).

Tests, in one run:
  [1] TYPE-B ENERGY: is misclassified CIFAR-10 energy-normal (DEV ~ correct), unlike iSUN?
  [2] TYPE-B SPLIT: on confident-wrong, does CLD fire while CONSENSUS stays low?
  [3] SEVERITY SWEEP: at severity 1/2/3, where does AUC leave the ceiling, and do the
      structure axes split at low severity?

Needs (from extract --typeb --sev-sweep --diag --cifar10c):
  diag_cifar10_correct, diag_cifar10_wrong, diag_cifar10_confwrong,
  diag_<corruption>_s1/_s2/_s3, diag_clean

Usage:
  python -m analysis.typeb_real --cache /workspace/cache/run9
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


def dir_auc(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    s = np.r_[a, b]
    if len(np.unique(s)) < 2:
        return 0.5
    return max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))


def step1_typeb_energy(diag):
    print("=" * 72)
    print("[1] TYPE-B ENERGY -- is misclassified CIFAR-10 energy-normal? (vs iSUN far-OOD)")
    print("=" * 72)
    di = AXES.index("DEVIATION")
    ref = diag.get("cifar10_correct", diag.get("clean"))
    if ref is None:
        print("  no correct/clean reference")
        return
    ref_dev = ref[:, di].mean()
    print(f"\n  reference (correct/clean) DEVIATION: {ref_dev:.3f}")
    for k in ["cifar10_wrong", "cifar10_confwrong", "isun"]:
        if k in diag:
            dev = diag[k][:, di].mean()
            print(f"  {k:22s} DEV={dev:.3f}  (vs ref {dev-ref_dev:+.3f})")
    print("\n  -> type-b should be energy-NORMAL (DEV near correct). If cifar10_wrong is")
    print("     close to correct but iSUN is far below, type-b != far-OOD (good).")


def step2_typeb_split(diag):
    print("\n" + "=" * 72)
    print("[2] TYPE-B SPLIT -- does confident-wrong split CLD from CONSENSUS?")
    print("=" * 72)
    print("type-b signature: CLUSTER_DISTANCE fires, CONSENSUS silent (votes don't split).")
    print("Reference = correctly-classified CIFAR-10.\n")
    ref = diag.get("cifar10_correct")
    if ref is None:
        print("  no diag_cifar10_correct reference")
        return
    print(f"  {'type-b set':24s} " + " ".join(f"{a[:8]:>9s}" for a in AXES) + "  struct-spread")
    print("  " + "-" * 82)
    for k in ["cifar10_wrong", "cifar10_confwrong"]:
        if k not in diag:
            continue
        aucs = [dir_auc(ref[:, i], diag[k][:, i]) for i in range(len(AXES))]
        struct = [aucs[AXES.index(s)] for s in STRUCT]
        spread = max(struct) - min(struct)
        print(f"  {k:24s} " + " ".join(f"{a:9.2f}" for a in aucs) + f"  {spread:.2f}")

    if "cifar10_confwrong" in diag:
        a = [dir_auc(ref[:, i], diag["cifar10_confwrong"][:, i]) for i in range(len(AXES))]
        cld = a[AXES.index("CLUSTER_DISTANCE")]
        con = a[AXES.index("CONSENSUS")]
        print(f"\n  conf-wrong: CLUSTER_DISTANCE={cld:.2f}, CONSENSUS={con:.2f}")
        if cld > 0.6 and cld > con + 0.15:
            print("  >>> TYPE-B SPLIT CONFIRMED: CLD fires, CONSENSUS lags -> CLD is the")
            print("      unique type-b detector -> structure axes are genuinely separate, KEEP CLD.")
        elif cld < 0.6 and con < 0.6:
            print("  >>> NEITHER fires on type-b -> structure axes MISS type-b entirely")
            print("      -> a NEW axis may be needed for row 4 (the real gap).")
        else:
            print("  >>> both fire similarly -> no split; structure axes track together even")
            print("      on type-b (evidence toward redundancy).")


def step3_severity_sweep(diag):
    print("\n" + "=" * 72)
    print("[3] SEVERITY SWEEP -- escape the AUC ceiling, look for splits at low severity")
    print("=" * 72)
    if "clean" not in diag:
        print("  no diag_clean")
        return
    clean = diag["clean"]
    # find corruptions with _s1/_s2/_s3
    corrs = sorted(set(k.rsplit("_s", 1)[0] for k in diag if "_s" in k and k[-1].isdigit()))
    if not corrs:
        print("  no severity-sweep caches (run extract --sev-sweep)")
        return
    for corr in corrs:
        print(f"\n  {corr}:")
        print(f"    {'sev':>4s} " + " ".join(f"{a[:8]:>9s}" for a in AXES) + "  struct-spread")
        for sev in (1, 2, 3):
            k = f"{corr}_s{sev}"
            if k not in diag:
                continue
            aucs = [dir_auc(clean[:, i], diag[k][:, i]) for i in range(len(AXES))]
            struct = [aucs[AXES.index(s)] for s in STRUCT]
            spread = max(struct) - min(struct)
            flag = " <-- below ceiling" if max(aucs) < 0.99 else ""
            split = " SPLIT" if spread > 0.2 else ""
            print(f"    {sev:4d} " + " ".join(f"{a:9.2f}" for a in aucs)
                  + f"  {spread:.2f}{split}{flag}")
    print("\n  -> at severities where AUC < 1.00 (below ceiling), a structure-axis spread")
    print("     >0.2 means the axes SPLIT -> not redundant. If they stay together even")
    print("     below ceiling -> evidence toward redundancy (collapse 7->5).")


def main(args):
    cache = FrameCache(args.cache)
    diag = load_diag(cache)
    print(f"diagnostic caches: {list(diag)}\n")
    has_typeb = "cifar10_wrong" in diag or "cifar10_confwrong" in diag
    has_sweep = any("_s" in k and k[-1].isdigit() for k in diag)
    if not has_typeb and not has_sweep:
        print("Neither type-b nor severity-sweep caches found.")
        print("Run: extract --typeb --sev-sweep --diag --cifar10c")
        return
    if has_typeb:
        step1_typeb_energy(diag)
        step2_typeb_split(diag)
    if has_sweep:
        step3_severity_sweep(diag)
    print("\n" + "=" * 72)
    print("DONE -- type-b (real) + severity-sweep analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
