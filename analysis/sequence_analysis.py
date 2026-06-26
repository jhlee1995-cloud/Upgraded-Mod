"""
analysis/sequence_analysis.py -- multi-step roadmap analysis in ONE run.

Runs on the run5 cache (point + stream + ramp caches). No GPU. Produces, in sequence:
  [1] 7-axis Exp0: covariance off-diagonal gate combining single-batch + sequence axes
  [2] CLUST_DRIFT unique-coverage: does it catch anything PERSIST+DRIFT_COH miss?
  [3] full axis correlation matrix (all axes together)
  [4] per-axis disturbance-response summary (which axis fires on what)

Usage:
  python -m analysis.sequence_analysis --cache /workspace/cache/run5
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache
from experiments.exp0 import sweep_curve, verdict
from sklearn.metrics import roc_auc_score


SEQ_COLS = ["DRIFT_COH_signed", "DRIFT_COH_abs", "PERSIST",
            "CLUST_DRIFT_signed", "CLUST_DRIFT_abs"]
POINT_AXES = ["DEVIATION", "CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS"]


def load_all(cache):
    """Gather every cache, grouped by kind."""
    point, stream, ramp = {}, {}, {}
    for name in cache.list_caches():
        if name.startswith("point_"):
            point[name[6:]] = cache._load(name)[0]
        elif name.startswith("stream_"):
            stream[name[7:]] = cache._load(name)[0]
        elif name.startswith("ramp_"):
            ramp[name[5:]] = cache._load(name)[0]
    return point, stream, ramp


def step1_7axis_exp0(point, stream):
    """TRUE 7-axis Exp0: stream caches now hold 9 columns (4 point-axis means +
    5 sequence) on ONE sampling unit (per stream). Combine the 4 point axes + 3
    meaningful sequence axes (DRIFT_COH_signed, PERSIST, CLUST_DRIFT_signed) = 7,
    run the off-diagonal covariance gate. Does the plateau hold with all 7?"""
    print("=" * 72)
    print("[1] 7-AXIS EXP0 -- joint covariance gate (point + sequence, same unit)")
    print("=" * 72)
    taus = np.linspace(0, 0.95, 20)

    # stream caches: columns = 4 point axes + [DRIFT_COH_s, DRIFT_COH_a, PERSIST,
    # CLUST_DRIFT_s, CLUST_DRIFT_a]. Pick 7: point[0:4] + signed seq [4,6,7].
    cols9 = POINT_AXES + SEQ_COLS
    if not stream:
        print("  no stream caches found")
        return
    sample = next(iter(stream.values()))
    if sample.shape[1] < 9:
        print(f"  stream caches have {sample.shape[1]} cols (old format, 5); "
              f"re-extract with updated extractor for joint 7-axis. Falling back to "
              f"separate point/sequence gates.")
        seven_idx = None
    else:
        # 7 axes: 0,1,2,3 (point) + 4 (DRIFT_COH_s), 6 (PERSIST), 7 (CLUST_DRIFT_s)
        seven_idx = [0, 1, 2, 3, 4, 6, 7]
        seven_names = POINT_AXES + ["DRIFT_COH", "PERSIST", "CLUST_DRIFT"]

    if seven_idx is not None:
        V7 = np.vstack(list(stream.values()))[:, seven_idx]
        # also pool ramp if present (same format)
        c7, _, mo7 = sweep_curve(V7, taus)
        print(f"\n7-axis joint gate: {V7.shape}, max|offdiag|={mo7:.2f}")
        print(f"VERDICT: {verdict(taus, c7)}")
        C = np.corrcoef(V7, rowvar=False)
        print("\n7-axis correlation matrix:")
        print("            " + " ".join(f"{n[:8]:>9s}" for n in seven_names))
        for i, n in enumerate(seven_names):
            print(f"{n[:11]:12s}" + " ".join(f"{C[i,j]:9.2f}" for j in range(7)))
        # two-group reading
        print("\ngrouping: which axes correlate? (|corr|>0.4 pairs)")
        for i in range(7):
            for j in range(i+1, 7):
                if abs(C[i, j]) > 0.4:
                    print(f"  {seven_names[i]} <-> {seven_names[j]}: {C[i,j]:+.2f}")
    else:
        # fallback: separate gates (old 5-col format)
        Vp = np.vstack(list(point.values()))
        cp, _, mop = sweep_curve(Vp, taus)
        print(f"\n(fallback a) point 4-axis: {Vp.shape}, max|offdiag|={mop:.2f}, "
              f"{verdict(taus, cp)}")
        Vs = np.vstack(list(stream.values()))[:, [0, 2, 3]]
        cs, _, mos = sweep_curve(Vs, taus)
        print(f"(fallback b) sequence 3-axis: {Vs.shape}, max|offdiag|={mos:.2f}, "
              f"{verdict(taus, cs)}")


def step2_clustdrift_coverage(stream, ramp):
    """Does CLUST_DRIFT catch anything PERSIST + DRIFT_COH miss?
    For each stream type, check if CLUST_DRIFT separates conditions that the other
    two don't. If CLUST_DRIFT never uniquely separates, it's covered (droppable)."""
    print("\n" + "=" * 72)
    print("[2] CLUST_DRIFT UNIQUE-COVERAGE TEST")
    print("=" * 72)
    print("Question: does CLUST_DRIFT separate any condition-pair that PERSIST and")
    print("DRIFT_COH both fail to separate? If never -> it's redundant (droppable).\n")

    # build a table: for each corruption, condition pairs (block vs shuffle,
    # ramp vs block), measure separation (|mean diff| / pooled std) per axis.
    def sep(a, b):
        return abs(a.mean() - b.mean()) / (np.sqrt(a.std()**2 + b.std()**2) + 1e-9)

    axes = {"DRIFT_COH": 0, "PERSIST": 2, "CLUST_DRIFT": 3}  # signed forms
    corruptions = set(k.rsplit("_", 1)[0] for k in stream if k.endswith("block"))

    print(f"{'pair':35s} {'DRIFT_COH':>10s} {'PERSIST':>9s} {'CLUST_DRIFT':>12s} {'CD unique?'}")
    print("-" * 85)
    clustdrift_ever_unique = False
    for corr in sorted(corruptions):
        pairs = []
        if f"{corr}_block" in stream and f"{corr}_shuffle" in stream:
            pairs.append(("block vs shuffle", stream[f"{corr}_block"], stream[f"{corr}_shuffle"]))
        if f"{corr}_linear" in ramp and f"{corr}_block" in stream:
            pairs.append(("ramp vs block", ramp[f"{corr}_linear"], stream[f"{corr}_block"]))
        for pname, A, B in pairs:
            seps = {ax: sep(A[:, i], B[:, i]) for ax, i in axes.items()}
            # CLUST_DRIFT "unique" if it separates (>1.0) where both others don't (<0.5)
            cd_unique = (seps["CLUST_DRIFT"] > 1.0 and
                         seps["DRIFT_COH"] < 0.5 and seps["PERSIST"] < 0.5)
            if cd_unique:
                clustdrift_ever_unique = True
            mark = "YES <--" if cd_unique else "no"
            print(f"{corr+' '+pname:35s} {seps['DRIFT_COH']:10.2f} "
                  f"{seps['PERSIST']:9.2f} {seps['CLUST_DRIFT']:12.2f} {mark}")

    print()
    if clustdrift_ever_unique:
        print(">>> CLUST_DRIFT uniquely separates at least one pair -> KEEP it.")
    else:
        print(">>> CLUST_DRIFT NEVER uniquely separates -> it's covered by PERSIST+DRIFT_COH.")
        print("    Candidate to DROP (pool 7 -> 6). Confirm on more disturbance types first.")


def step3_full_correlation(point):
    """Full correlation matrix of the single-batch axes (the ones that share a
    sampling unit). Re-confirms the two-group structure on run5."""
    print("\n" + "=" * 72)
    print("[3] SINGLE-BATCH AXIS CORRELATION (run5)")
    print("=" * 72)
    V = np.vstack(list(point.values()))
    C = np.corrcoef(V, rowvar=False)
    print("        " + " ".join(f"{a[:8]:>9s}" for a in POINT_AXES))
    for i, a in enumerate(POINT_AXES):
        print(f"{a[:7]:8s}" + " ".join(f"{C[i,j]:9.2f}" for j in range(len(POINT_AXES))))
    # report the two-group reading
    dev_row = C[0, 1:]
    print(f"\nDEVIATION vs others: {dev_row.round(2)}  "
          f"({'orthogonal/negative' if dev_row.mean()<0 else 'positive'})")


def main(args):
    cache = FrameCache(args.cache)
    point, stream, ramp = load_all(cache)
    print(f"cache: {args.cache} | synthetic={cache.is_synthetic()}")
    print(f"point caches: {list(point)}")
    print(f"stream caches: {list(stream)}")
    print(f"ramp caches: {list(ramp)}\n")

    step1_7axis_exp0(point, stream)
    step2_clustdrift_coverage(stream, ramp)
    step3_full_correlation(point)
    print("\n" + "=" * 72)
    print("DONE -- 3 analyses complete on cached data.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
