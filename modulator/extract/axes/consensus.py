"""
AXIS: CONSENSUS / SUBNET-CONSENSUS -- agreement among K channel-group subnets.

Channel-group subnets (WIDTH split, not depth -- depth heads are dead, AUC~0.5).
Each subnet 'votes' on which clean center the input is nearest. Disagreement =
low consensus = anomaly.

Ideal behavior (from the known finding): CONSENSUS fires on
  - structure_break (groups pushed apart -> disagree)
  - cluster_a (type-a, between centers -> votes split)
and stays quiet on
  - energy (all groups scale together -> still agree)
  - cluster_b (type-b, toward ONE wrong center -> votes DON'T split; this is the
    fundamental blind spot -- only CLUSTER-DISTANCE catches type-b)

Each candidate: (activations, clean_ref) -> per-sample score (higher = less consensus).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

FEAT_DIM = 64
N_GROUPS = 4
GROUP = FEAT_DIM // N_GROUPS


class CleanRefC:
    """Per-subnet clean class centers (each subnet sees only its channel slice)."""
    def __init__(self, clean, labels, n_classes=10):
        self.n_classes = n_classes
        self.subnet_centers = []  # [group][class] -> center in that group's subspace
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            centers = np.zeros((n_classes, GROUP))
            for c in range(n_classes):
                pts = clean[labels == c, sl]
                centers[c] = pts.mean(0) if len(pts) else 0
            self.subnet_centers.append(centers)
        # clean-time baseline disagreement (for normalization)
        self._baseline = None

    def subnet_votes(self, X):
        """For each subnet, nearest-center class index per sample -> (n, N_GROUPS)."""
        n = X.shape[0]
        votes = np.zeros((n, N_GROUPS), dtype=int)
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            Xg = X[:, sl]
            d = np.linalg.norm(Xg[:, None] - self.subnet_centers[g][None], axis=2)
            votes[:, g] = d.argmin(axis=1)
        return votes

    def subnet_logits(self, X):
        """Soft 'logit' = negative distance to each center, per subnet -> (n, G, C)."""
        n = X.shape[0]
        L = np.zeros((n, N_GROUPS, self.n_classes))
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            Xg = X[:, sl]
            d = np.linalg.norm(Xg[:, None] - self.subnet_centers[g][None], axis=2)
            L[:, g] = -d
        return L


# ----------------------------------------------------------------------------
# candidate formulas (higher = less consensus = more anomalous)
# ----------------------------------------------------------------------------
def cons_disagree_rate(X, ref):
    """(a) fraction of subnet pairs that vote DIFFERENTLY."""
    votes = ref.subnet_votes(X)
    n = X.shape[0]
    score = np.zeros(n)
    pairs = 0
    for g1 in range(N_GROUPS):
        for g2 in range(g1 + 1, N_GROUPS):
            score += (votes[:, g1] != votes[:, g2])
            pairs += 1
    return score / pairs

def cons_majority_gap(X, ref):
    """(b) 1 - (fraction of subnets agreeing with the majority vote)."""
    votes = ref.subnet_votes(X)
    n = X.shape[0]
    out = np.zeros(n)
    for i in range(n):
        vals, counts = np.unique(votes[i], return_counts=True)
        out[i] = 1.0 - counts.max() / N_GROUPS
    return out

def cons_logit_variance(X, ref):
    """(c) variance of per-subnet predicted class (soft): how spread are the
    subnets' preferred classes? Uses logit argmax dispersion via center spread."""
    L = ref.subnet_logits(X)              # (n, G, C)
    pred = L.argmax(axis=2)               # (n, G)
    return pred.std(axis=1)

def cons_vote_entropy(X, ref):
    """(d) entropy of the vote distribution across subnets (high = split)."""
    votes = ref.subnet_votes(X)
    n = X.shape[0]
    out = np.zeros(n)
    for i in range(n):
        _, counts = np.unique(votes[i], return_counts=True)
        p = counts / counts.sum()
        out[i] = -(p * np.log(p + 1e-9)).sum()
    return out


CANDIDATES = {
    "disagree_rate":  cons_disagree_rate,
    "majority_gap":   cons_majority_gap,
    "logit_variance": cons_logit_variance,
    "vote_entropy":   cons_vote_entropy,
}


def cross_reaction(data, labels, ref_name="clean"):
    clean = data[ref_name]
    ref = CleanRefC(clean, labels)
    disturbances = [k for k in data if k != ref_name]
    print(f"{'candidate':16s} | " + " ".join(f"{d[:10]:>11s}" for d in disturbances))
    print("-" * (18 + 12 * len(disturbances)))
    rows = {}
    for cname, fn in CANDIDATES.items():
        s_clean = fn(clean, ref)
        aucs = []
        for d in disturbances:
            s_dist = fn(data[d], ref)
            y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_dist))]
            s = np.r_[s_clean, s_dist]
            # guard against degenerate (all-equal) scores
            auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
            aucs.append(auc)
        rows[cname] = dict(zip(disturbances, aucs))
        print(f"{cname:16s} | " + " ".join(f"{a:11.2f}" for a in aucs))
    return rows


def score_selectivity(rows, targets=("structure", "cluster_a"),
                      non_targets=("energy", "cluster_b")):
    """CONSENSUS ideal: high on structure AND cluster_a; ~0.5 on energy/cluster_b."""
    print(f"\n{'candidate':16s} | {'tgt_mean':>9s} {'nontgt_leak':>12s} {'selectivity':>12s}")
    print("-" * 54)
    best, best_score = None, -1
    for cname, aucs in rows.items():
        tgt = np.mean([aucs[t] for t in targets])
        leak = np.mean([abs(aucs[nt] - 0.5) for nt in non_targets]) * 2
        sel = tgt - leak
        if sel > best_score:
            best_score, best = sel, cname
        print(f"{cname:16s} | {tgt:9.2f} {leak:12.2f} {sel:12.2f}")
    print(f"\n  BEST CONSENSUS formula: {best}  (selectivity {best_score:.2f})")
    print(f"  (note: cluster_b SHOULD be ~0.5 -- type-b blind spot is correct, not a bug)")
    return best


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/modulator/extract")
    from synth_activations import generate
    data, centers, lab = generate()
    print("=" * 70)
    print("CONSENSUS axis -- candidate formula cross-reaction")
    print("=" * 70)
    print("(ideal: high on structure & cluster_a, ~0.5 on energy & cluster_b)\n")
    rows = cross_reaction(data, lab)
    best = score_selectivity(rows)
