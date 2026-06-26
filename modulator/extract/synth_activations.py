"""
Synthetic activation generator -- for testing 6-axis extraction FORMULAS.

Produces (batch, feat_dim) activations like ResNet20 penultimate (feat_dim=64):
  clean            : 10-class structure (multi-center)
  energy_burst     : DEVIATION should catch (norm inflation)
  structure_break  : CONSENSUS should catch (channel-group disagreement)
  cluster_shift_a  : SUBNET-CONSENSUS should catch (type-a, between centers)
  cluster_shift_b  : CLUSTER-DISTANCE should catch (type-b, wrong center, norm-preserved)
  + mixtures       : test SELECTIVITY (does each axis respond to its own, ignore others?)

type-b carries the Stage-B caveat: even at activation level, norm-preserving
displacement to a wrong center may leak; flagged synthetic-unreliable and
deferred to Stage A real near-OOD where needed.

This is logic-validation only; real activations come from Stage A on RunPod.
"""
import numpy as np

FEAT_DIM = 64
N_CLASSES = 10
N_GROUPS = 4          # channel-group subnets (width split, not depth)
GROUP = FEAT_DIM // N_GROUPS


def _class_centers(rng, sep=6.0):
    return rng.standard_normal((N_CLASSES, FEAT_DIM)) * sep


def clean_batch(centers, n, rng, spread=1.0):
    """n activations drawn from the 10-class constellation."""
    labels = rng.integers(0, N_CLASSES, n)
    X = centers[labels] + rng.standard_normal((n, FEAT_DIM)) * spread
    return X, labels


def energy_burst(X, rng, factor=2.5):
    """norm inflation: scale activations up (DEVIATION target)."""
    return X * factor


def structure_break(X, centers, labels, rng, strength=12.0):
    """break channel-group coherence: each group gets an INDEPENDENT random push
    LARGER than the inter-center distance, so the groups (subnets) actually flip
    their votes and disagree. (A push smaller than center-separation leaves votes
    unchanged -- correct CONSENSUS behavior is to ignore sub-threshold breaks.)"""
    Y = X.copy()
    for g in range(N_GROUPS):
        sl = slice(g * GROUP, (g + 1) * GROUP)
        Y[:, sl] += rng.standard_normal((X.shape[0], GROUP)) * strength
    return Y


def cluster_shift_a(X, centers, labels, rng):
    """type-a: move each point to the MIDPOINT between its center and a random other
    center -> belongs to neither (between clusters). Energy roughly preserved."""
    Y = X.copy()
    for i in range(X.shape[0]):
        c0 = labels[i]
        c1 = rng.integers(0, N_CLASSES)
        while c1 == c0:
            c1 = rng.integers(0, N_CLASSES)
        mid = 0.5 * (centers[c0] + centers[c1])
        Y[i] = mid + rng.standard_normal(FEAT_DIM) * 1.0
    return Y


def cluster_shift_b(X, centers, labels, rng):
    """type-b: move each point NEAR a WRONG center, with norm explicitly preserved
    (the crux: energy must look normal). SYNTHETIC-UNRELIABLE -- see caveat."""
    Y = X.copy()
    for i in range(X.shape[0]):
        c0 = labels[i]
        c1 = rng.integers(0, N_CLASSES)
        while c1 == c0:
            c1 = rng.integers(0, N_CLASSES)
        target = centers[c1] + rng.standard_normal(FEAT_DIM) * 1.0
        # preserve original norm to keep energy "normal"
        Y[i] = target / (np.linalg.norm(target) + 1e-9) * np.linalg.norm(X[i])
    return Y


DISTURBANCES = {
    "energy":      lambda X, c, l, r: energy_burst(X, r),
    "structure":   lambda X, c, l, r: structure_break(X, c, l, r),
    "cluster_a":   lambda X, c, l, r: cluster_shift_a(X, c, l, r),
    "cluster_b":   lambda X, c, l, r: cluster_shift_b(X, c, l, r),
}
SYNTHETIC_UNRELIABLE = {"cluster_b"}  # type-b: defer to Stage A real near-OOD


def generate(n=2000, seed=0):
    """Return dict: 'clean', each pure disturbance, and pairwise mixtures.
    Also returns the centers (the frame) and clean labels for reference."""
    rng = np.random.default_rng(seed)
    centers = _class_centers(rng)
    Xc, lab = clean_batch(centers, n, rng)

    out = {"clean": Xc}
    # pure
    for name, fn in DISTURBANCES.items():
        out[name] = fn(Xc.copy(), centers, lab, rng)
    # mixtures (apply two disturbances in sequence)
    mixes = [("energy", "structure"), ("energy", "cluster_b"),
             ("structure", "cluster_a"), ("energy", "cluster_a"),
             ("structure", "cluster_b")]
    for a, b in mixes:
        Y = DISTURBANCES[a](Xc.copy(), centers, lab, rng)
        Y = DISTURBANCES[b](Y, centers, lab, rng)
        out[f"{a}+{b}"] = Y

    return out, centers, lab


if __name__ == "__main__":
    data, centers, lab = generate()
    print("synthetic activations generated:")
    for k, v in data.items():
        flag = "  [synthetic-unreliable]" if any(s in k for s in SYNTHETIC_UNRELIABLE) else ""
        print(f"  {k:22s}: {v.shape}  mean-norm={np.linalg.norm(v, axis=1).mean():6.2f}{flag}")
