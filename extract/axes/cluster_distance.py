"""
AXIS: SUBNET-CLUSTER-DISTANCE -- mean normalized distance to nearest class center
across K channel-group subnets. THE type-b detector.

type-b (cluster_b) is toward a WRONG center with energy preserved: invisible to
DEVIATION (energy normal) and CONSENSUS (votes don't split -- all subnets agree
on the wrong center). The ONLY axis that catches it is distance: the input sits
far from its TRUE center even though it's near A center.

Wait -- if type-b is NEAR a (wrong) center, nearest-center distance is SMALL.
So naive nearest-distance does NOT catch type-b either. The signal must use the
distance pattern that distinguishes "near the right center" (clean) from "near a
center but displaced" -- candidates explore which distance statistic exposes it.

Ideal: fires on cluster_b, and ALSO on cluster_a (between centers -> far from all)
and structure (groups scattered). Quiet on energy (scales but stays near center
after per-subnet normalization).

Includes a diagnostic: if type-b is missed, sweep its strength to tell
formula-defect from weak-synthetic (the lesson from DEVIATION/CONSENSUS).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

FEAT_DIM = 64
N_GROUPS = 4
GROUP = FEAT_DIM // N_GROUPS


class CleanRefD:
    def __init__(self, clean, labels, n_classes=10):
        self.n_classes = n_classes
        self.subnet_centers = []
        self.subnet_scale = []   # per-subnet typical within-cluster distance (normalizer)
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            centers = np.zeros((n_classes, GROUP))
            for c in range(n_classes):
                pts = clean[labels == c, sl]
                centers[c] = pts.mean(0) if len(pts) else 0
            self.subnet_centers.append(centers)
            # scale = median nearest-center distance of clean points in this subnet
            Xg = clean[:, sl]
            d = np.linalg.norm(Xg[:, None] - centers[None], axis=2).min(1)
            self.subnet_scale.append(np.median(d) + 1e-9)

    def per_subnet_nearest(self, X):
        """(n, N_GROUPS) nearest-center distance per subnet, normalized by subnet scale."""
        n = X.shape[0]
        out = np.zeros((n, N_GROUPS))
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            Xg = X[:, sl]
            d = np.linalg.norm(Xg[:, None] - self.subnet_centers[g][None], axis=2).min(1)
            out[:, g] = d / self.subnet_scale[g]
        return out

    def per_subnet_all(self, X):
        """(n, N_GROUPS, n_classes) distance to EVERY center per subnet (normalized)."""
        n = X.shape[0]
        out = np.zeros((n, N_GROUPS, self.n_classes))
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            Xg = X[:, sl]
            d = np.linalg.norm(Xg[:, None] - self.subnet_centers[g][None], axis=2)
            out[:, g] = d / self.subnet_scale[g]
        return out


# ----------------------------------------------------------------------------
# candidate formulas (higher = more anomalous)
# ----------------------------------------------------------------------------
def dist_mean_nearest(X, ref):
    """(a) mean over subnets of nearest-center distance."""
    return ref.per_subnet_nearest(X).mean(axis=1)

def dist_max_nearest(X, ref):
    """(b) max over subnets of nearest-center distance."""
    return ref.per_subnet_nearest(X).max(axis=1)

def dist_disagreement(X, ref):
    """(c) do subnets disagree on WHICH center is nearest, AND how far?
    type-b: subnets may pick different wrong centers -> distance-vector spread."""
    alld = ref.per_subnet_all(X)          # (n, G, C)
    nearest_idx = alld.argmin(axis=2)     # (n, G)
    # spread of nearest-center choices across subnets
    n = X.shape[0]
    spread = np.array([len(np.unique(nearest_idx[i])) for i in range(n)])
    return spread.astype(float)

def dist_second_margin(X, ref):
    """(d) margin between nearest and 2nd-nearest center, averaged over subnets.
    clean: large margin (clearly one center). type-b near wrong center: also large.
    type-a between centers: SMALL margin. (catches type-a, maybe not type-b.)"""
    alld = ref.per_subnet_all(X)          # (n, G, C)
    sorted_d = np.sort(alld, axis=2)
    margin = sorted_d[:, :, 1] - sorted_d[:, :, 0]   # (n, G)
    return -margin.mean(axis=1)           # small margin -> high score

def dist_true_center(X, ref, true_labels):
    """(e) ORACLE: distance to the TRUE center (needs labels -- only for diagnosis).
    type-b is far from its TRUE center by construction. This shows the CEILING:
    what's detectable IF we knew the true label. Real label-free axes can't use this."""
    n = X.shape[0]
    out = np.zeros(n)
    for g in range(N_GROUPS):
        sl = slice(g * GROUP, (g + 1) * GROUP)
        Xg = X[:, sl]
        for i in range(n):
            c = true_labels[i]
            d = np.linalg.norm(Xg[i] - ref.subnet_centers[g][c]) / ref.subnet_scale[g]
            out[i] += d
    return out / N_GROUPS


CANDIDATES = {
    "mean_nearest":   dist_mean_nearest,
    "max_nearest":    dist_max_nearest,
    "disagreement":   dist_disagreement,
    "second_margin":  dist_second_margin,
}


def cross_reaction(data, labels, ref_name="clean"):
    clean = data[ref_name]
    ref = CleanRefD(clean, labels)
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
            auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
            aucs.append(auc)
        rows[cname] = dict(zip(disturbances, aucs))
        print(f"{cname:16s} | " + " ".join(f"{a:11.2f}" for a in aucs))
    return rows, ref


def diagnose_typeb(data, labels, ref):
    """If candidates miss type-b, is it formula-defect or weak-synthetic?
    Show: (1) oracle true-center distance AUC (the detectable ceiling),
           (2) how 'wrong' the synthetic type-b actually is."""
    clean = data["clean"]
    tb = data["cluster_b"]
    print("\n  --- type-b diagnosis ---")
    # oracle: distance to TRUE center (clean labels apply to tb since tb was built from clean)
    s_clean = dist_true_center(clean, ref, labels)
    s_tb = dist_true_center(tb, ref, labels)
    y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_tb))]
    s = np.r_[s_clean, s_tb]
    oracle_auc = roc_auc_score(y, s)
    print(f"  oracle (true-center dist) AUC = {oracle_auc:.2f}  "
          f"(ceiling: detectable IF label known)")
    print(f"  best label-free candidate on cluster_b = "
          f"{max(roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(tb))], np.r_[fn(clean, ref), fn(tb, ref)]) for fn in CANDIDATES.values()):.2f}")
    if oracle_auc > 0.8:
        print("  -> type-b IS displaced (oracle catches it). If label-free misses it,")
        print("     that's the KNOWN hard problem -> needs real near-OOD at Stage A.")
    else:
        print("  -> synthetic type-b is too weak (even oracle misses) -> strengthen synth.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/modulator/extract")
    from synth_activations import generate
    data, centers, lab = generate()
    print("=" * 70)
    print("SUBNET-CLUSTER-DISTANCE axis -- candidate cross-reaction")
    print("=" * 70)
    print("(ideal: fires on cluster_b [type-b], also cluster_a/structure; quiet on energy)\n")
    rows, ref = cross_reaction(data, lab)
    diagnose_typeb(data, lab, ref)
