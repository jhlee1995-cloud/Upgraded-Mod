"""
frame/coordinate.py -- multi-center coordinate frame (Stage B brain, validated).

Clean activations define the RULER (multi-center constellation); only misalignment
is drawn as branches. Built once, time-invariant (the fixed-anchor principle).

Validated on synthetic ground truth (see STAGE_B_RESULTS.md):
  - k-sweep recovers the planted number of clean clusters
  - severity-monotone distance (frame encodes misalignment strength)
  - out-of-frame mass tracks disturbance ratio
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_sweep(X, ks, n_seeds=20):
    """silhouette vs k over seeds; the plateau/peak = number of clean clusters.
    Returns (k_hat, mean_silhouette_per_k, std_per_k)."""
    means, stds = [], []
    for k in ks:
        s = []
        for seed in range(n_seeds):
            km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
            s.append(silhouette_score(X, km.labels_) if k > 1 else 0.0)
        means.append(np.mean(s)); stds.append(np.std(s))
    means, stds = np.array(means), np.array(stds)
    k_hat = list(ks)[int(np.argmax(means))]
    return k_hat, means, stds


class CoordinateFrame:
    """Multi-center constellation with per-cluster Mahalanobis metric."""

    def __init__(self, clean, k):
        self.k = k
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(clean)
        self.centers = km.cluster_centers_
        labels = km.labels_
        self.inv_covs, self.covs = [], []
        D = clean.shape[1]
        for c in range(k):
            pts = clean[labels == c]
            if len(pts) > D + 1:
                cov = np.cov(pts, rowvar=False) + 1e-3 * np.eye(D)
            else:
                cov = np.eye(D)
            self.covs.append(cov)
            self.inv_covs.append(np.linalg.inv(cov))
        # frame scale: typical clean nearest-center Mahalanobis distance
        d = self.mahalanobis_nearest(clean)
        self.thr = float(np.percentile(d, 99))  # out-of-frame threshold

    @classmethod
    def build(cls, clean, ks=range(2, 16)):
        """Build via k-sweep (recovers cluster count automatically)."""
        k_hat, means, stds = k_sweep(clean, list(ks))
        frame = cls(clean, k_hat)
        frame.k_sweep_means = means
        frame.k_sweep_ks = list(ks)
        return frame

    def mahalanobis_nearest(self, X):
        n, k = X.shape[0], self.k
        d = np.zeros((n, k))
        for c in range(k):
            diff = X - self.centers[c]
            d[:, c] = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_covs[c], diff))
        return d.min(axis=1)

    def euclidean_nearest(self, X):
        return np.min(np.linalg.norm(X[:, None] - self.centers[None], axis=2), axis=1)

    # --- branch measurement, two ways ---
    def out_of_frame_mass(self, X):
        """(b) fraction of X beyond the frame's known region (Mahalanobis)."""
        return float(np.mean(self.mahalanobis_nearest(X) > self.thr))

    def added_clusters(self, X, k_extra_range=range(0, 8), n_seeds=5,
                       min_points=60, min_per_cluster=25):
        """(a) #clusters formed by out-of-frame points (min-points gate vs noise)."""
        d = self.mahalanobis_nearest(X)
        outside = X[d > self.thr]
        if len(outside) < min_points:
            return 0
        best_k, best_sil = 1, -1
        for k in k_extra_range:
            if k < 2 or k >= len(outside):
                continue
            if len(outside) / k < min_per_cluster:
                break
            sils = [silhouette_score(outside, KMeans(n_clusters=k, n_init=3,
                    random_state=s).fit(outside).labels_) for s in range(n_seeds)]
            if np.mean(sils) > best_sil:
                best_sil, best_k = np.mean(sils), k
        return best_k if best_sil > 0.3 else 1
