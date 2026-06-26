"""
Synthetic STREAM generator -- plants TIME structure to logic-validate sequence axes.

Sequence axes (DRIFT_COH, PERSIST, CLUST_DRIFT) need distribution MOVEMENT over time
that static synthetic data lacks (the Module-4 wall). Here we plant time structure:

  clean_stream       : distribution fixed over T          (no drift/persist signal)
  drift_stream       : distribution shifts one direction  (DRIFT_COH should fire)
  persistent_anomaly : anomaly sustained across all T      (PERSIST high)
  transient_anomaly  : anomaly in ONE step only            (PERSIST low)
  clust_drift_stream : distance-vector shifts structurally (CLUST_DRIFT should fire)

A stream = list of T feature batches, each (B, feat_dim). Built from the same
10-class constellation as the single-batch synth.
"""
import numpy as np

FEAT_DIM = 64
N_CLASSES = 10


def _centers(rng, sep=6.0):
    return rng.standard_normal((N_CLASSES, FEAT_DIM)) * sep


def _clean_batch(centers, n, rng, spread=1.0):
    lab = rng.integers(0, N_CLASSES, n)
    return centers[lab] + rng.standard_normal((n, FEAT_DIM)) * spread, lab


def clean_stream(centers, T, B, rng):
    """distribution fixed: each step is an independent clean draw."""
    return [_clean_batch(centers, B, rng)[0] for _ in range(T)]


def drift_stream(centers, T, B, rng, drift_mag=20.0):
    """distribution mean shifts steadily in ONE fixed direction across T steps.
    drift_mag must exceed batch-noise spread or the directional trend is buried
    (correct: sub-noise drift is genuinely undetectable in real time)."""
    direction = rng.standard_normal(FEAT_DIM)
    direction /= np.linalg.norm(direction)
    stream = []
    for t in range(T):
        X, _ = _clean_batch(centers, B, rng)
        X = X + direction * (drift_mag * t / max(T - 1, 1))  # 0 -> drift_mag
        stream.append(X)
    return stream


def persistent_anomaly(centers, T, B, rng, energy=2.5):
    """anomaly (energy inflation) present in EVERY step."""
    return [_clean_batch(centers, B, rng)[0] * energy for _ in range(T)]


def transient_anomaly(centers, T, B, rng, energy=2.5, spike_at=None):
    """anomaly in ONE step only; the rest clean."""
    if spike_at is None:
        spike_at = T // 2
    stream = []
    for t in range(T):
        X, _ = _clean_batch(centers, B, rng)
        if t == spike_at:
            X = X * energy
        stream.append(X)
    return stream


def clust_drift_stream(centers, T, B, rng, shift_mag=4.0):
    """the K-dim distance-vector shifts structurally: move points progressively
    toward a fixed WRONG center, so per-subnet distances change coherently while
    the overall energy stays moderate (distinct from a plain mean drift)."""
    wrong = centers[rng.integers(N_CLASSES)]
    stream = []
    for t in range(T):
        X, lab = _clean_batch(centers, B, rng)
        frac = t / max(T - 1, 1)
        X = (1 - 0.6 * frac) * X + (0.6 * frac) * wrong  # creep toward wrong center
        stream.append(X)
    return stream


STREAM_TYPES = {
    "clean":             clean_stream,
    "drift":             drift_stream,
    "persistent":        persistent_anomaly,
    "transient":         transient_anomaly,
    "clust_drift":       clust_drift_stream,
}


def generate_streams(n_streams=40, T=8, B=64, seed=0):
    """Return dict: stream_type -> list of n_streams streams (each a list of T (B,64))."""
    rng = np.random.default_rng(seed)
    centers = _centers(rng)
    out = {}
    for name, fn in STREAM_TYPES.items():
        streams = []
        for s in range(n_streams):
            r = np.random.default_rng(seed * 1000 + s)
            streams.append(fn(centers, T, B, r))
        out[name] = streams
    return out, centers


if __name__ == "__main__":
    streams, centers = generate_streams(n_streams=3)
    print("synthetic streams generated (time structure planted):")
    for name, sl in streams.items():
        s0 = sl[0]
        # show energy trajectory over T to confirm planted structure
        energies = [float((b ** 2).mean()) for b in s0]
        print(f"  {name:12s}: T={len(s0)} B={s0[0].shape[0]}  "
              f"energy traj = [{', '.join(f'{e:.0f}' for e in energies)}]")
