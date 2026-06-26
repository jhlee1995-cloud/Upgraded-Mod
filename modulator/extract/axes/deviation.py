"""
AXIS: DEVIATION -- normalized activation-energy deviation from clean.

Tests MULTIPLE candidate formulas and measures SELECTIVITY: DEVIATION should
fire on energy_burst and stay quiet on structure / cluster_a / cluster_b.
The cross-reaction row (AUC of each candidate vs clean, per disturbance) reveals
whether DEVIATION is orthogonal to the other axes' targets.

Each candidate: (activations, clean_ref) -> per-sample score (higher = more deviant).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------
# clean reference statistics (fit once on clean activations)
# ----------------------------------------------------------------------------
class CleanRef:
    def __init__(self, clean):
        self.norms = np.linalg.norm(clean, axis=1)
        self.norm_mean = self.norms.mean()
        self.norm_std = self.norms.std() + 1e-9
        self.norm_med = np.median(self.norms)
        self.norm_mad = np.median(np.abs(self.norms - self.norm_med)) + 1e-9
        # per-channel energy
        self.chan_energy_mean = (clean ** 2).mean(0)
        self.chan_energy_std = (clean ** 2).std(0) + 1e-9


# ----------------------------------------------------------------------------
# candidate formulas
# ----------------------------------------------------------------------------
def dev_l2_zscore(X, ref):
    """(a) |L2 norm - clean mean| / clean std."""
    n = np.linalg.norm(X, axis=1)
    return np.abs(n - ref.norm_mean) / ref.norm_std

def dev_l2_robust(X, ref):
    """(b) robust z-score using median / MAD (outlier-resistant)."""
    n = np.linalg.norm(X, axis=1)
    return np.abs(n - ref.norm_med) / ref.norm_mad

def dev_l2_signed(X, ref):
    """(c) signed deviation (only inflation counts; deflation -> 0)."""
    n = np.linalg.norm(X, axis=1)
    return np.maximum(0, (n - ref.norm_mean) / ref.norm_std)

def dev_per_channel(X, ref):
    """(d) aggregate per-channel energy deviation (mean over channels)."""
    e = X ** 2
    z = np.abs(e - ref.chan_energy_mean) / ref.chan_energy_std
    return z.mean(axis=1)

def dev_per_channel_max(X, ref):
    """(e) per-channel energy deviation, MAX over channels (catches single-channel spikes)."""
    e = X ** 2
    z = np.abs(e - ref.chan_energy_mean) / ref.chan_energy_std
    return z.max(axis=1)


CANDIDATES = {
    "l2_zscore":      dev_l2_zscore,
    "l2_robust":      dev_l2_robust,
    "l2_signed":      dev_l2_signed,
    "per_channel":    dev_per_channel,
    "per_channel_max": dev_per_channel_max,
}


# ----------------------------------------------------------------------------
# cross-reaction test
# ----------------------------------------------------------------------------
def cross_reaction(data, ref_name="clean"):
    """For each candidate, AUC(clean vs each disturbance).
    DEVIATION ideal: ~1.0 on energy, ~0.5 on structure/cluster_a/cluster_b."""
    clean = data[ref_name]
    ref = CleanRef(clean)
    disturbances = [k for k in data if k != ref_name]

    print(f"{'candidate':18s} | " + " ".join(f"{d[:10]:>11s}" for d in disturbances))
    print("-" * (20 + 12 * len(disturbances)))
    rows = {}
    for cname, fn in CANDIDATES.items():
        s_clean = fn(clean, ref)
        aucs = []
        for d in disturbances:
            s_dist = fn(data[d], ref)
            y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_dist))]
            s = np.r_[s_clean, s_dist]
            auc = roc_auc_score(y, s)
            aucs.append(auc)
        rows[cname] = dict(zip(disturbances, aucs))
        print(f"{cname:18s} | " + " ".join(f"{a:11.2f}" for a in aucs))
    return rows


def score_selectivity(rows, target="energy"):
    """A good DEVIATION formula: high AUC on target, ~0.5 on non-targets.
    Selectivity = AUC(target) - mean(|AUC(non-target) - 0.5|)*2  (penalize cross-reaction)."""
    print(f"\n{'candidate':18s} | {'target_auc':>10s} {'cross_leak':>11s} {'selectivity':>12s}")
    print("-" * 56)
    best, best_score = None, -1
    for cname, aucs in rows.items():
        target_auc = aucs[target]
        non_targets = [v for k, v in aucs.items() if target not in k]
        cross_leak = np.mean([abs(v - 0.5) for v in non_targets]) * 2
        sel = target_auc - cross_leak
        if sel > best_score:
            best_score, best = sel, cname
        print(f"{cname:18s} | {target_auc:10.2f} {cross_leak:11.2f} {sel:12.2f}")
    print(f"\n  BEST DEVIATION formula: {best}  (selectivity {best_score:.2f})")
    return best


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/modulator/extract")
    from synth_activations import generate

    data, centers, lab = generate()
    print("=" * 70)
    print("DEVIATION axis -- candidate formula cross-reaction")
    print("=" * 70)
    print("(ideal: ~1.0 on 'energy', ~0.5 on structure/cluster_a/cluster_b)\n")
    rows = cross_reaction(data)
    best = score_selectivity(rows, target="energy")
