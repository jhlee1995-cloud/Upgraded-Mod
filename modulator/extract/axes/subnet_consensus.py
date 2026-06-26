"""
AXIS: SUBNET-CONSENSUS (6th) vs CONSENSUS (2nd) -- complementarity test.

CONSENSUS (2nd, hard vote): do subnets agree on WHICH center is nearest? (argmax)
SUBNET-CONSENSUS (6th, soft): do subnets agree on the DISTRIBUTION / confidence?

These should diverge when subnets agree on the center but disagree on confidence
(hard vote identical, soft distribution spread). We add a disturbance that creates
exactly this -- 'confidence_split' -- and check whether the two axes separate.

  redundant   : 6th tracks 2nd on every disturbance (same row) -> they're one axis
  complementary: 6th fires on confidence_split where 2nd is quiet -> distinct axes

(known claim: "neither subsumes the other" -- verified here, not assumed.)
"""
import numpy as np
from sklearn.metrics import roc_auc_score

FEAT_DIM = 64
N_GROUPS = 4
GROUP = FEAT_DIM // N_GROUPS


class SubnetRef:
    def __init__(self, clean, labels, n_classes=10):
        self.n_classes = n_classes
        self.subnet_centers = []
        self.subnet_scale = []
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            centers = np.zeros((n_classes, GROUP))
            for c in range(n_classes):
                pts = clean[labels == c, sl]
                centers[c] = pts.mean(0) if len(pts) else 0
            self.subnet_centers.append(centers)
            Xg = clean[:, sl]
            d = np.linalg.norm(Xg[:, None] - centers[None], axis=2).min(1)
            self.subnet_scale.append(np.median(d) + 1e-9)

    def soft_probs(self, X, temp=1.0):
        """(n, G, C) softmax over -distance per subnet = soft class distribution."""
        n = X.shape[0]
        P = np.zeros((n, N_GROUPS, self.n_classes))
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            Xg = X[:, sl]
            d = np.linalg.norm(Xg[:, None] - self.subnet_centers[g][None], axis=2)
            logits = -d / (self.subnet_scale[g] * temp)
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            P[:, g] = e / e.sum(axis=1, keepdims=True)
        return P

    def hard_votes(self, X):
        n = X.shape[0]
        votes = np.zeros((n, N_GROUPS), dtype=int)
        for g in range(N_GROUPS):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            Xg = X[:, sl]
            d = np.linalg.norm(Xg[:, None] - self.subnet_centers[g][None], axis=2)
            votes[:, g] = d.argmin(axis=1)
        return votes


# --- the 2nd axis (hard vote) for contrast ---
def consensus_hard(X, ref):
    """CONSENSUS: pairwise hard-vote disagreement rate."""
    votes = ref.hard_votes(X)
    n = X.shape[0]
    score = np.zeros(n); pairs = 0
    for g1 in range(N_GROUPS):
        for g2 in range(g1 + 1, N_GROUPS):
            score += (votes[:, g1] != votes[:, g2]); pairs += 1
    return score / pairs

# --- the 6th axis candidates (soft) ---
def subcons_js_divergence(X, ref):
    """(a) mean pairwise Jensen-Shannon divergence between subnet distributions."""
    P = ref.soft_probs(X)                  # (n, G, C)
    n = X.shape[0]
    out = np.zeros(n); pairs = 0
    for g1 in range(N_GROUPS):
        for g2 in range(g1 + 1, N_GROUPS):
            p, q = P[:, g1], P[:, g2]
            m = 0.5 * (p + q)
            kl_pm = (p * (np.log(p + 1e-9) - np.log(m + 1e-9))).sum(1)
            kl_qm = (q * (np.log(q + 1e-9) - np.log(m + 1e-9))).sum(1)
            out += 0.5 * (kl_pm + kl_qm); pairs += 1
    return out / pairs

def subcons_confidence_var(X, ref):
    """(b) variance across subnets of the max-probability (confidence)."""
    P = ref.soft_probs(X)
    conf = P.max(axis=2)                    # (n, G)
    return conf.std(axis=1)

def subcons_mean_entropy(X, ref):
    """(c) mean per-subnet entropy (how uncertain each subnet is, averaged)."""
    P = ref.soft_probs(X)
    ent = -(P * np.log(P + 1e-9)).sum(axis=2)   # (n, G)
    return ent.mean(axis=1)


SUBCONS_CANDIDATES = {
    "js_divergence":   subcons_js_divergence,
    "confidence_var":  subcons_confidence_var,
    "mean_entropy":    subcons_mean_entropy,
}


def confidence_split(X, centers, labels, rng, jitter=2.0):
    """NEW disturbance: keep each subnet voting for the SAME (correct) center,
    but make subnets disagree on CONFIDENCE -- push each subnet a different
    fraction toward its center boundary WITHOUT crossing it. Hard vote unchanged,
    soft distribution spread. This is where 2nd and 6th should diverge."""
    Y = X.copy()
    for g in range(N_GROUPS):
        sl = slice(g * GROUP, (g + 1) * GROUP)
        # per-subnet, per-sample random scale of the within-cluster offset
        scale = rng.uniform(0.2, 1.0, (X.shape[0], 1)) * jitter
        # move toward/away from the subnet center along current offset (no vote flip)
        cen = np.zeros((X.shape[0], GROUP))
        for i in range(X.shape[0]):
            cen[i] = centers[labels[i], g * GROUP:(g + 1) * GROUP]
        offset = Y[:, sl] - cen
        Y[:, sl] = cen + offset * (1 + scale)   # stretch offset -> lower confidence, same argmax
    return Y


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/modulator/extract")
    from synth_activations import generate, _class_centers, clean_batch

    data, centers, lab = generate()
    # add the confidence_split disturbance built from the SAME clean batch
    rng = np.random.default_rng(7)
    data["confidence_split"] = confidence_split(data["clean"].copy(), centers, lab, rng)

    ref = SubnetRef(data["clean"], lab)
    disturbances = [k for k in data if k != "clean"]

    print("=" * 74)
    print("SUBNET-CONSENSUS (6th, soft) vs CONSENSUS (2nd, hard) -- complementarity")
    print("=" * 74)
    print("(key column: 'confidence_split' -- 2nd should be ~0.5, 6th should fire)\n")

    clean = data["clean"]
    # 2nd axis row
    s_clean = consensus_hard(clean, ref)
    print(f"{'2nd:consensus_hard':22s} | " + " ".join(f"{d[:9]:>10s}" for d in disturbances))
    print("-" * (24 + 11 * len(disturbances)))
    row2 = []
    for d in disturbances:
        s = np.r_[s_clean, consensus_hard(data[d], ref)]
        y = np.r_[np.zeros(len(clean)), np.ones(len(data[d]))]
        auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
        row2.append(auc)
    print(f"{'2nd:consensus_hard':22s} | " + " ".join(f"{a:10.2f}" for a in row2))

    # 6th axis rows
    print()
    rows6 = {}
    for cname, fn in SUBCONS_CANDIDATES.items():
        s_clean6 = fn(clean, ref)
        aucs = []
        for d in disturbances:
            s = np.r_[s_clean6, fn(data[d], ref)]
            y = np.r_[np.zeros(len(clean)), np.ones(len(data[d]))]
            auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
            aucs.append(auc)
        rows6[cname] = dict(zip(disturbances, aucs))
        print(f"{'6th:'+cname:22s} | " + " ".join(f"{a:10.2f}" for a in aucs))

    # complementarity verdict
    print("\n  --- complementarity verdict ---")
    cs_idx = disturbances.index("confidence_split")
    print(f"  2nd (hard) on confidence_split: {row2[cs_idx]:.2f}  (expect ~0.5 = blind)")
    for cname, aucs in rows6.items():
        v = aucs["confidence_split"]
        verdict = "COMPLEMENTARY" if (v > 0.7 and row2[cs_idx] < 0.65) else "redundant?"
        print(f"  6th ({cname}) on confidence_split: {v:.2f}  -> {verdict}")
