"""
Sequence-axis candidates -- tested on planted-time-structure streams.

Three sequence axes, multiple candidate formulas each. We measure selectivity:
each axis should fire on ITS time structure and stay quiet on others. Key question:
do DRIFT_COH (mean-push coherence) and CLUST_DRIFT (distance-vector coherence)
DIVERGE, or are they redundant?

Each candidate: (stream = [h_1..h_T], ref) -> scalar score.
Validation: AUC separating clean streams from each structured stream.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

FEAT_DIM = 64
N_GROUPS = 4
GROUP = FEAT_DIM // N_GROUPS


class StreamRef:
    def __init__(self, clean_streams, centers):
        self.centers = centers
        # per-subnet class centers from clean stream batches
        allh = np.vstack([b for s in clean_streams for b in s])
        # crude labels via nearest full-dim center (clean is near its own center)
        d = np.linalg.norm(allh[:, None] - centers[None], axis=2)
        lab = d.argmin(1)
        self.subnet_centers = []
        self.subnet_scale = []
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            cs = np.zeros((len(centers), GROUP))
            for c in range(len(centers)):
                m = lab == c
                cs[c] = allh[m][:, sl].mean(0) if m.any() else allh[:, sl].mean(0)
            self.subnet_centers.append(cs)
            dd = np.linalg.norm(allh[:, sl][:, None] - cs[None], axis=2).min(1)
            self.subnet_scale.append(np.median(dd) + 1e-9)
        self.energy_mean = (allh ** 2).mean()

    def dist_vector(self, h):
        """K-dim per-subnet mean nearest-center distance for a batch -> (K,)."""
        out = np.zeros(N_GROUPS)
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            d = np.linalg.norm(h[:, sl][:, None] - self.subnet_centers[g][None], axis=2).min(1)
            out[g] = (d / self.subnet_scale[g]).mean()
        return out


# ---------------------------------------------------------------------------
# DRIFT_COH candidates (mean-push direction coherence)
# ---------------------------------------------------------------------------
def drift_consec_cosine(stream, ref):
    """(a) [prev notebook] mean cosine of consecutive mean-push directions."""
    means = np.array([h.mean(0) for h in stream])
    push = np.diff(means, axis=0)
    pn = push / (np.linalg.norm(push, axis=1, keepdims=True) + 1e-9)
    if len(pn) < 2:
        return 0.0
    return float(np.mean([np.dot(pn[i], pn[i + 1]) for i in range(len(pn) - 1)]))

def drift_path_straightness(stream, ref):
    """(b) straightness = |end - start| / total path length (1 = straight drift)."""
    means = np.array([h.mean(0) for h in stream])
    seg = np.linalg.norm(np.diff(means, axis=0), axis=1).sum()
    net = np.linalg.norm(means[-1] - means[0])
    return float(net / (seg + 1e-9))

def drift_window_cosine(stream, ref, w=2):
    """(c) coherence of windowed-average pushes (smoother)."""
    means = np.array([h.mean(0) for h in stream])
    if len(means) < 2 * w + 1:
        return drift_consec_cosine(stream, ref)
    wm = np.array([means[i:i + w].mean(0) for i in range(len(means) - w)])
    push = np.diff(wm, axis=0)
    pn = push / (np.linalg.norm(push, axis=1, keepdims=True) + 1e-9)
    if len(pn) < 2:
        return 0.0
    return float(np.mean([np.dot(pn[i], pn[i + 1]) for i in range(len(pn) - 1)]))


# ---------------------------------------------------------------------------
# PERSIST candidates (sustained vs transient)
# ---------------------------------------------------------------------------
def persist_autocorr(stream, ref):
    """(a) [prev notebook] lag-1 autocorrelation of energy deviation."""
    devs = np.array([(h ** 2).mean() / ref.energy_mean - 1.0 for h in stream])
    d = devs - devs.mean()
    return float((d[1:] * d[:-1]).sum() / ((d * d).sum() + 1e-9))

def persist_streak(stream, ref, thr=0.5):
    """(b) longest consecutive run of above-threshold energy deviation / T."""
    devs = np.array([(h ** 2).mean() / ref.energy_mean - 1.0 for h in stream])
    over = devs > thr
    best = cur = 0
    for o in over:
        cur = cur + 1 if o else 0
        best = max(best, cur)
    return float(best / len(stream))

def persist_fraction(stream, ref, thr=0.5):
    """(c) fraction of steps above threshold (persistent=high, transient=low)."""
    devs = np.array([(h ** 2).mean() / ref.energy_mean - 1.0 for h in stream])
    return float(np.mean(devs > thr))


# ---------------------------------------------------------------------------
# CLUST_DRIFT candidates (distance-vector change coherence)
# ---------------------------------------------------------------------------
def clustdrift_consec_cosine(stream, ref):
    """(a) [prev notebook] coherence of consecutive distance-vector changes."""
    dv = np.array([ref.dist_vector(h) for h in stream])
    ch = np.diff(dv, axis=0)
    cn = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    if len(cn) < 2:
        return 0.0
    return float(np.mean([np.dot(cn[i], cn[i + 1]) for i in range(len(cn) - 1)]))

def clustdrift_total_shift(stream, ref):
    """(b) net displacement of the distance-vector start->end."""
    dv = np.array([ref.dist_vector(h) for h in stream])
    return float(np.linalg.norm(dv[-1] - dv[0]))

def clustdrift_straightness(stream, ref):
    """(c) straightness of the distance-vector trajectory."""
    dv = np.array([ref.dist_vector(h) for h in stream])
    seg = np.linalg.norm(np.diff(dv, axis=0), axis=1).sum()
    net = np.linalg.norm(dv[-1] - dv[0])
    return float(net / (seg + 1e-9))


AXIS_CANDIDATES = {
    "DRIFT_COH": {
        "consec_cosine":   drift_consec_cosine,
        "path_straight":   drift_path_straightness,
        "window_cosine":   drift_window_cosine,
    },
    "PERSIST": {
        "autocorr":        persist_autocorr,
        "streak":          persist_streak,
        "fraction":        persist_fraction,
    },
    "CLUST_DRIFT": {
        "consec_cosine":   clustdrift_consec_cosine,
        "total_shift":     clustdrift_total_shift,
        "straightness":    clustdrift_straightness,
    },
}

# which stream each axis SHOULD fire on
AXIS_TARGET = {
    "DRIFT_COH":   "drift",
    "PERSIST":     "persistent",
    "CLUST_DRIFT": "clust_drift",
}


def auc_vs_clean(fn, ref, clean_streams, test_streams):
    s_clean = np.array([fn(s, ref) for s in clean_streams])
    s_test = np.array([fn(s, ref) for s in test_streams])
    y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_test))]
    s = np.r_[s_clean, s_test]
    return roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/modulator/extract")
    from synth_streams import generate_streams

    streams, centers = generate_streams(n_streams=40)
    ref = StreamRef(streams["clean"], centers)
    stream_types = [k for k in streams if k != "clean"]

    print("=" * 78)
    print("SEQUENCE AXES -- candidate cross-reaction on planted-time-structure streams")
    print("=" * 78)

    chosen = {}
    for axis, cands in AXIS_CANDIDATES.items():
        target = AXIS_TARGET[axis]
        print(f"\n[{axis}]  target stream = '{target}'")
        print(f"  {'candidate':16s} | " + " ".join(f"{t[:11]:>12s}" for t in stream_types))
        print("  " + "-" * (18 + 13 * len(stream_types)))
        best, best_sel = None, -1e9
        for cname, fn in cands.items():
            aucs = {t: auc_vs_clean(fn, ref, streams["clean"], streams[t]) for t in stream_types}
            tgt = aucs[target]
            leak = np.mean([abs(aucs[t] - 0.5) for t in stream_types if t != target]) * 2
            sel = tgt - leak
            # PERSIST special: its real job is persistent-VS-transient, not vs clean
            extra = ""
            if axis == "PERSIST":
                s_p = np.array([fn(s, ref) for s in streams["persistent"]])
                s_t = np.array([fn(s, ref) for s in streams["transient"]])
                y = np.r_[np.zeros(len(s_t)), np.ones(len(s_p))]
                s = np.r_[s_t, s_p]
                pvt = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
                sel = pvt  # the metric that matters for PERSIST
                extra = f"  [persist-vs-transient AUC={pvt:.2f}]"
            if sel > best_sel:
                best_sel, best = sel, cname
            print(f"  {cname:16s} | " + " ".join(f"{aucs[t]:12.2f}" for t in stream_types)
                  + f"   sel={sel:.2f}{extra}")
        chosen[axis] = best
        print(f"  -> best: {best} (selectivity {best_sel:.2f})")

    # ---- the key question: DRIFT_COH vs CLUST_DRIFT redundancy ----
    print("\n" + "=" * 78)
    print("DRIFT_COH vs CLUST_DRIFT -- redundant or complementary?")
    print("=" * 78)
    dfn = AXIS_CANDIDATES["DRIFT_COH"][chosen["DRIFT_COH"]]
    cfn = AXIS_CANDIDATES["CLUST_DRIFT"][chosen["CLUST_DRIFT"]]
    print(f"  using DRIFT_COH={chosen['DRIFT_COH']}, CLUST_DRIFT={chosen['CLUST_DRIFT']}")
    print(f"  {'stream':14s} | {'DRIFT_COH':>10s} {'CLUST_DRIFT':>12s}")
    for t in ["drift", "clust_drift"]:
        da = auc_vs_clean(dfn, ref, streams["clean"], streams[t])
        ca = auc_vs_clean(cfn, ref, streams["clean"], streams[t])
        print(f"  {t:14s} | {da:10.2f} {ca:12.2f}")
    print("\n  complementary IF: DRIFT_COH fires on 'drift' but not 'clust_drift',")
    print("                    CLUST_DRIFT fires on 'clust_drift' but not 'drift'.")
    print("  redundant IF:     both fire on both equally.")
