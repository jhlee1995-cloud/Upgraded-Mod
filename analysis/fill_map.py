"""
analysis/fill_map.py -- fill as many Error-Coverage-Map cells as possible in one pass.

Reads every stream cache (block/shuffle, ramp, recovery) and reports, per map row, the
primary metric and whether the cell should be ✅ (real signal) or ✗ (tested, none).
Rows addressed here (the temporal + sustained block):
  row 5 directional drift      -- ramp vs block (DRIFT_COH)
  row 6 non-directional drift  -- does PERSIST catch motion_blur ramp where DRIFT_COH fails?
  row 7 sustained              -- block vs shuffle (PERSIST)
  row 9 recovery               -- does any axis read DIRECTION (up vs down)?

(rows 2/3/4 + valleys are covered by structure_geometry / typeb_real / structure_redundancy.)

Usage:
  python -m analysis.fill_map --cache /workspace/cache/run11
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame.cache import FrameCache

# 10-col joint stream format (4 point + 6 sequence incl TRAJ_LOOP)
COLS = ["DEVIATION", "CONSENSUS", "CLUSTER_DISTANCE", "SUBNET_CONSENSUS",
        "DRIFT_COH_signed", "DRIFT_COH_abs", "PERSIST",
        "CLUST_DRIFT_signed", "CLUST_DRIFT_abs", "TRAJ_LOOP"]
IDX = {c: i for i, c in enumerate(COLS)}


def sep(a, b):
    return abs(a.mean() - b.mean()) / (np.sqrt(a.std()**2 + b.std()**2) + 1e-9)


def load_streams(cache):
    streams, ramps, recovs = {}, {}, {}
    for name in cache.list_caches():
        if name.startswith("stream_"):
            streams[name[7:]] = cache._load(name)[0]
        elif name.startswith("ramp_"):
            ramps[name[5:]] = cache._load(name)[0]
        elif name.startswith("recovery_"):
            recovs[name[9:]] = cache._load(name)[0]
    return streams, ramps, recovs


def row5_directional_drift(streams, ramps):
    print("=" * 72)
    print("[row 5] DIRECTIONAL DRIFT -- ramp vs block (metric: DRIFT_COH separation)")
    print("=" * 72)
    corrs = sorted(set(k.rsplit("_", 1)[0] for k in streams if k.endswith("block")))
    any_sig = False
    for corr in corrs:
        rk = f"{corr}_linear"
        bk = f"{corr}_block"
        if rk in ramps and bk in streams:
            s = sep(ramps[rk][:, IDX["DRIFT_COH_signed"]], streams[bk][:, IDX["DRIFT_COH_signed"]])
            dirn = "directional" if s > 1.0 else "weak/non-dir"
            print(f"  {corr:16s} DRIFT_COH ramp-vs-block sep = {s:5.2f}  ({dirn})")
            if s > 1.0:
                any_sig = True
    print(f"\n  -> DRIFT_COH catches directional drift where sep>1 (e.g. gaussian). "
          f"{'✅ signal present' if any_sig else '✗ none'}")


def row6_nondirectional_drift(streams, ramps):
    print("\n" + "=" * 72)
    print("[row 6] NON-DIRECTIONAL DRIFT -- does PERSIST catch what DRIFT_COH misses?")
    print("=" * 72)
    print("For corruptions where DRIFT_COH is blind on ramp (non-directional, e.g. motion_blur),")
    print("does PERSIST still separate ramp from clean-ish? metric: PERSIST on ramp vs block.\n")
    corrs = sorted(set(k.rsplit("_", 1)[0] for k in streams if k.endswith("block")))
    for corr in corrs:
        rk = f"{corr}_linear"
        bk = f"{corr}_block"
        if rk in ramps and bk in streams:
            dch = sep(ramps[rk][:, IDX["DRIFT_COH_signed"]], streams[bk][:, IDX["DRIFT_COH_signed"]])
            per = sep(ramps[rk][:, IDX["PERSIST"]], streams[bk][:, IDX["PERSIST"]])
            verdict = ""
            if dch < 1.0 and per > 1.0:
                verdict = " <-- PERSIST covers what DRIFT_COH misses (✅ row 6)"
            elif dch < 1.0 and per < 1.0:
                verdict = " <-- BOTH weak: row 6 gap (propose new axis?)"
            print(f"  {corr:16s} DRIFT_COH sep={dch:5.2f}  PERSIST sep={per:5.2f}{verdict}")


def row7_sustained(streams):
    print("\n" + "=" * 72)
    print("[row 7] SUSTAINED -- block vs shuffle (metric: PERSIST separation)")
    print("=" * 72)
    corrs = sorted(set(k.rsplit("_", 1)[0] for k in streams if k.endswith("block")))
    any_sig = False
    for corr in corrs:
        bk, sk = f"{corr}_block", f"{corr}_shuffle"
        if bk in streams and sk in streams:
            b, s = streams[bk][:, IDX["PERSIST"]], streams[sk][:, IDX["PERSIST"]]
            sp = sep(b, s)
            print(f"  {corr:16s} PERSIST block {b.mean():.3f}±{b.std():.3f} vs "
                  f"shuffle {s.mean():.3f}±{s.std():.3f}  (sep {sp:.2f})")
            if sp > 1.0:
                any_sig = True
    print(f"\n  -> {'✅ PERSIST separates sustained' if any_sig else '✗ none'}")


def row9_recovery(recovs, ramps, streams):
    print("\n" + "=" * 72)
    print("[row 9] RECOVERY -- TRAJ_LOOP (Takens-PH global return structure)")
    print("=" * 72)
    if not recovs:
        print("  no recovery caches (run extract --recovery)")
        return
    print("recovery goes up THEN back down (returns); ramp/block only go up (monotone).")
    print("TRAJ_LOOP = loop content of the Takens-embedded trajectory. Validated synthetic:")
    print("recovery loops (H1~1.07), drift/sustained don't (~0). Compare recovery vs ramp:\n")
    print(f"  {'corruption':16s} {'TRAJ_LOOP recov':>15s} {'TRAJ_LOOP ramp':>15s} "
          f"{'TRAJ_LOOP block':>15s}  verdict")
    print("  " + "-" * 80)
    ti = IDX["TRAJ_LOOP"]
    any_sig = False
    for corr in sorted(recovs):
        rl = recovs[corr][:, ti].mean()
        ml = ramps.get(f"{corr}_linear", np.zeros((1, len(COLS))))[:, ti].mean()
        bl = streams.get(f"{corr}_block", np.zeros((1, len(COLS))))[:, ti].mean()
        sep_ramp = sep(recovs[corr][:, ti], ramps[f"{corr}_linear"][:, ti]) \
            if f"{corr}_linear" in ramps else 0.0
        verdict = ""
        if rl > ml + 0.2 and rl > bl + 0.2:
            verdict = f" recovery LOOPS (sep vs ramp {sep_ramp:.1f}) ✅"
            any_sig = True
        else:
            verdict = " no clear loop separation"
        print(f"  {corr:16s} {rl:15.3f} {ml:15.3f} {bl:15.3f} {verdict}")
    print()
    if any_sig:
        print("  >>> ✅ TRAJ_LOOP separates recovery (returns) from ramp/block (monotone).")
        print("      This is the direction/return signal DRIFT_COH (local cosine) cannot see.")
        print("      Map row 9 (recovery) -> TRAJ_LOOP is a real detector.")
    else:
        print("  >>> TRAJ_LOOP did not separate recovery here -- check trajectory length /")
        print("      tau; or recovery signal too noisy on real data.")
    # signed drift for reference (does it also flip?)
    print(f"\n  reference DRIFT_COH_signed: " + " ".join(
        f"{c}={recovs[c][:, IDX['DRIFT_COH_signed']].mean():+.2f}" for c in sorted(recovs)))


def main(args):
    cache = FrameCache(args.cache)
    streams, ramps, recovs = load_streams(cache)
    sample = next(iter(streams.values()), None)
    if sample is None or sample.shape[1] < 9:
        print("Need 9-col joint stream caches (re-extract with current code).")
        print(f"(streams: {list(streams)})")
        return
    print(f"streams: {list(streams)}\nramps: {list(ramps)}\nrecovery: {list(recovs)}\n")
    row5_directional_drift(streams, ramps)
    row6_nondirectional_drift(streams, ramps)
    row7_sustained(streams)
    row9_recovery(recovs, ramps, streams)
    print("\n" + "=" * 72)
    print("DONE -- temporal + sustained map rows filled. Promote cells ✅/✗ in the map")
    print("based on the separations above (sep>1 = ✅, sep~0 = ✗).")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    main(ap.parse_args())
