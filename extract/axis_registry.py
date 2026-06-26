"""
Axis registry -- the 7 chosen axis formulas behind ONE interface.

Single-batch axes : f(X, ref) -> per-sample score        (X is (B, C))
Sequence axes     : f(stream, ref) -> scalar             (stream is [X_1..X_T])

Chosen formulas (from synthetic validation, see *_RESULTS.md):
  single-batch: DEVIATION=l2_signed, CONSENSUS=disagree_rate,
                CLUSTER_DISTANCE=mean_nearest, SUBNET_CONSENSUS=mean_entropy
  sequence:     DRIFT_COH=window_cosine, PERSIST=streak, CLUST_DRIFT=consec_cosine

All redundancy questions (DRIFT_COH/CLUST_DRIFT, DISTANCE/DEVIATION) are kept OPEN
to Stage A -- both members of each suspected-redundant pair stay in the pool
(false-redundancy rule: never prune on synthetic).
"""
import numpy as np

# ---- subnet config (channel-group split; width not depth) ----
N_GROUPS = 4


class AxisRef:
    """Per-layer reference stats fit on clean activations of THAT layer.
    feat_dim is read from the layer; subnet group size = feat_dim / N_GROUPS."""

    def __init__(self, clean, labels, n_classes, n_groups=N_GROUPS):
        self.D = clean.shape[1]
        assert self.D % n_groups == 0, f"feat_dim {self.D} not divisible by {n_groups}"
        self.G = n_groups
        self.GS = self.D // n_groups
        self.n_classes = n_classes

        norms = np.linalg.norm(clean, axis=1)
        self.norm_mean = norms.mean()
        self.norm_std = norms.std() + 1e-9
        self.energy_mean = (clean ** 2).mean()

        # per-subnet class centers + scales
        self.subnet_centers, self.subnet_scale = [], []
        for g in range(self.G):
            sl = slice(g * self.GS, (g + 1) * self.GS)
            cs = np.zeros((n_classes, self.GS))
            for c in range(n_classes):
                m = labels == c
                cs[c] = clean[m][:, sl].mean(0) if m.any() else clean[:, sl].mean(0)
            self.subnet_centers.append(cs)
            d = np.linalg.norm(clean[:, sl][:, None] - cs[None], axis=2).min(1)
            self.subnet_scale.append(np.median(d) + 1e-9)

    # --- subnet primitives ---
    def per_subnet_nearest(self, X):
        out = np.zeros((X.shape[0], self.G))
        for g in range(self.G):
            sl = slice(g * self.GS, (g + 1) * self.GS)
            d = np.linalg.norm(X[:, sl][:, None] - self.subnet_centers[g][None], axis=2).min(1)
            out[:, g] = d / self.subnet_scale[g]
        return out

    def hard_votes(self, X):
        v = np.zeros((X.shape[0], self.G), dtype=int)
        for g in range(self.G):
            sl = slice(g * self.GS, (g + 1) * self.GS)
            d = np.linalg.norm(X[:, sl][:, None] - self.subnet_centers[g][None], axis=2)
            v[:, g] = d.argmin(1)
        return v

    def soft_probs(self, X, temp=1.0):
        P = np.zeros((X.shape[0], self.G, self.n_classes))
        for g in range(self.G):
            sl = slice(g * self.GS, (g + 1) * self.GS)
            d = np.linalg.norm(X[:, sl][:, None] - self.subnet_centers[g][None], axis=2)
            lg = -d / (self.subnet_scale[g] * temp)
            lg -= lg.max(1, keepdims=True)
            e = np.exp(lg)
            P[:, g] = e / e.sum(1, keepdims=True)
        return P

    def dist_vector(self, X):
        return self.per_subnet_nearest(X).mean(0)


# ============================================================
# SINGLE-BATCH AXES  f(X, ref) -> per-sample score
# ============================================================
def deviation(X, ref):
    """l2_signed: one-sided energy inflation (deflation -> 0). Avoids type-a leak."""
    n = np.linalg.norm(X, axis=1)
    return np.maximum(0, (n - ref.norm_mean) / ref.norm_std)

def consensus(X, ref):
    """disagree_rate: pairwise hard-vote disagreement among subnets."""
    v = ref.hard_votes(X)
    n, score, pairs = X.shape[0], np.zeros(X.shape[0]), 0
    for g1 in range(ref.G):
        for g2 in range(g1 + 1, ref.G):
            score += (v[:, g1] != v[:, g2]); pairs += 1
    return score / pairs

def cluster_distance(X, ref):
    """mean_nearest: mean per-subnet nearest-center distance. Sole (partial) type-b channel."""
    return ref.per_subnet_nearest(X).mean(1)

def subnet_consensus(X, ref):
    """mean_entropy: mean per-subnet soft entropy. Complementary to hard consensus."""
    P = ref.soft_probs(X)
    ent = -(P * np.log(P + 1e-9)).sum(axis=2)
    return ent.mean(axis=1)


# ============================================================
# SEQUENCE AXES  f(stream, ref) -> scalar
# ============================================================
# ============================================================
# AGGREGATE ANOMALY -- the per-batch scalar that PERSIST/DRIFT_COH ride on.
# Real-data finding: corruption LOWERS energy but RAISES distance, so the
# temporal axes must track an anomaly signal that actually moves with disturbance.
# Default = mean per-subnet nearest-center distance (validated: PERSIST(distance)
# caught sustained corruption 0.50 while PERSIST(energy)=0.00). Parameterized so it
# can later become a multi-axis or conformal-gate aggregate.
# ============================================================
def aggregate_anomaly(X, ref, mode="distance"):
    """Per-batch scalar anomaly. mode='distance' (validated default),
    'structure' (mean of the 3 structure axes), or 'energy' (legacy)."""
    if mode == "distance":
        return float(ref.per_subnet_nearest(X).mean())
    if mode == "structure":
        d = ref.per_subnet_nearest(X).mean()
        c = consensus(X, ref).mean()
        s = subnet_consensus(X, ref).mean()
        return float((d + c + s) / 3.0)
    if mode == "energy":
        return float((X ** 2).mean() / ref.energy_mean - 1.0)
    raise ValueError(mode)


# ============================================================
# SEQUENCE AXES -- each returns BOTH signed and absolute forms.
# Sign convention is unresolved on real data (CLUST_DRIFT came out NEGATIVE;
# "drift-coherence sign flips between domains"). Store both -> decide from data,
# one extraction instead of re-running after guessing the sign.
# ============================================================
def _coherence_pairs(vecs):
    """mean cosine of consecutive unit changes of a (T, D) trajectory.
    Returns the signed mean; caller takes abs if wanted."""
    ch = np.diff(vecs, axis=0)
    cn = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    if len(cn) < 2:
        return 0.0
    return float(np.mean([np.dot(cn[i], cn[i + 1]) for i in range(len(cn) - 1)]))


def drift_coh(stream, ref, w=2, signed=True):
    """window_cosine of mean-push directions (noise-robust). signed or absolute."""
    means = np.array([h.mean(0) for h in stream])
    if len(means) < 2 * w + 1:
        push = np.diff(means, axis=0)
    else:
        wm = np.array([means[i:i + w].mean(0) for i in range(len(means) - w)])
        push = np.diff(wm, axis=0)
    pn = push / (np.linalg.norm(push, axis=1, keepdims=True) + 1e-9)
    if len(pn) < 2:
        return 0.0
    val = float(np.mean([np.dot(pn[i], pn[i + 1]) for i in range(len(pn) - 1)]))
    return val if signed else abs(val)


def persist(stream, ref, mode="distance", k=2.0):
    """streak of consecutive ABOVE-BASELINE aggregate-anomaly batches / T.
    Re-based on aggregate_anomaly (not energy): threshold = baseline mean + k*std
    estimated from the first few batches (assumed cleaner). Real-data validated:
    distance-based catches sustained corruption where energy-based gives 0."""
    a = np.array([aggregate_anomaly(h, ref, mode=mode) for h in stream])
    n_base = max(2, len(a) // 3)
    base = a[:n_base].mean() + k * (a[:n_base].std() + 1e-9)
    over = a > base
    best = cur = 0
    for o in over:
        cur = cur + 1 if o else 0
        best = max(best, cur)
    return float(best / len(stream))


def clust_drift(stream, ref, signed=True):
    """consec_cosine of distance-VECTOR changes (K-dim). signed or absolute.
    Real-data: came out negative (distance-vector zig-zags) -> store both forms."""
    dv = np.array([ref.dist_vector(h) for h in stream])
    val = _coherence_pairs(dv)
    return val if signed else abs(val)


def compute_sequence_axes(stream, ref, mode="distance"):
    """Compute all sequence axes returning BOTH signed and absolute drift forms.
    Returns an ordered dict matching SEQUENCE_AXIS_COLUMNS below."""
    return {
        "DRIFT_COH_signed": drift_coh(stream, ref, signed=True),
        "DRIFT_COH_abs":    drift_coh(stream, ref, signed=False),
        "PERSIST":          persist(stream, ref, mode=mode),
        "CLUST_DRIFT_signed": clust_drift(stream, ref, signed=True),
        "CLUST_DRIFT_abs":    clust_drift(stream, ref, signed=False),
    }


SEQUENCE_AXIS_COLUMNS = ["DRIFT_COH_signed", "DRIFT_COH_abs", "PERSIST",
                         "CLUST_DRIFT_signed", "CLUST_DRIFT_abs"]



SINGLE_BATCH_AXES = {
    "DEVIATION": deviation,
    "CONSENSUS": consensus,
    "CLUSTER_DISTANCE": cluster_distance,
    "SUBNET_CONSENSUS": subnet_consensus,
}
# sequence axes are computed together via compute_sequence_axes (returns signed+abs
# drift forms); the individual fns remain for direct use/testing
SEQUENCE_AXES = {
    "DRIFT_COH": drift_coh,
    "PERSIST": persist,
    "CLUST_DRIFT": clust_drift,
}
ALL_AXES = list(SINGLE_BATCH_AXES) + SEQUENCE_AXIS_COLUMNS


if __name__ == "__main__":
    # smoke test on synthetic activations
    import sys
    sys.path.insert(0, "/home/claude/modulator/extract")
    from synth_activations import generate
    data, centers, lab = generate(n=800)
    ref = AxisRef(data["clean"], lab, n_classes=10)
    print("axis registry smoke test (synthetic clean):")
    for name, fn in SINGLE_BATCH_AXES.items():
        s = fn(data["clean"], ref)
        print(f"  {name:18s}: score shape {s.shape}, mean {s.mean():.3f}")
    # sequence axes need a stream
    from synth_streams import generate_streams
    streams, c2 = generate_streams(n_streams=2)
    allb = np.vstack([b for s in streams['clean'] for b in s])
    pseudo = np.linalg.norm(allb[:, None] - c2[None], axis=2).argmin(1)
    ref2 = AxisRef(allb, pseudo, n_classes=10)
    print("sequence axes (signed + abs forms):")
    seq = compute_sequence_axes(streams['clean'][0], ref2)
    for name, v in seq.items():
        print(f"  {name:20s}: {v:.3f}")
