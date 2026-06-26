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
def drift_coh(stream, ref, w=2):
    """window_cosine: coherence of windowed-average mean-pushes (noise-robust)."""
    means = np.array([h.mean(0) for h in stream])
    if len(means) < 2 * w + 1:
        push = np.diff(means, axis=0)
    else:
        wm = np.array([means[i:i + w].mean(0) for i in range(len(means) - w)])
        push = np.diff(wm, axis=0)
    pn = push / (np.linalg.norm(push, axis=1, keepdims=True) + 1e-9)
    if len(pn) < 2:
        return 0.0
    return float(np.mean([np.dot(pn[i], pn[i + 1]) for i in range(len(pn) - 1)]))

def persist(stream, ref, thr=0.5):
    """streak: longest consecutive run of above-threshold energy deviation / T."""
    devs = np.array([(h ** 2).mean() / ref.energy_mean - 1.0 for h in stream])
    over = devs > thr
    best = cur = 0
    for o in over:
        cur = cur + 1 if o else 0
        best = max(best, cur)
    return float(best / len(stream))

def clust_drift(stream, ref):
    """consec_cosine: coherence of consecutive distance-vector changes."""
    dv = np.array([ref.dist_vector(h) for h in stream])
    ch = np.diff(dv, axis=0)
    cn = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    if len(cn) < 2:
        return 0.0
    return float(np.mean([np.dot(cn[i], cn[i + 1]) for i in range(len(cn) - 1)]))


SINGLE_BATCH_AXES = {
    "DEVIATION": deviation,
    "CONSENSUS": consensus,
    "CLUSTER_DISTANCE": cluster_distance,
    "SUBNET_CONSENSUS": subnet_consensus,
}
SEQUENCE_AXES = {
    "DRIFT_COH": drift_coh,
    "PERSIST": persist,
    "CLUST_DRIFT": clust_drift,
}
ALL_AXES = list(SINGLE_BATCH_AXES) + list(SEQUENCE_AXES)


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
    ref2 = AxisRef(np.vstack([b for s in streams['clean'] for b in s]),
                   np.zeros(sum(len(b) for s in streams['clean'] for b in s), dtype=int)
                   if False else
                   __import__('numpy').linalg.norm(
                       np.vstack([b for s in streams['clean'] for b in s])[:, None] - c2[None], axis=2).argmin(1),
                   n_classes=10)
    for name, fn in SEQUENCE_AXES.items():
        v = fn(streams['clean'][0], ref2)
        print(f"  {name:18s}: scalar {v:.3f}")
