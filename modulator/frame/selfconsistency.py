"""
frame/selfconsistency.py -- tuning vs missed-axis discriminator (validated).

Bake the frame on train-historical, feed held-out back; a new cluster's
separability in axis-space vs raw-activation-space discriminates:
  separable in axis-space  -> TUNING (clustering failed; fix k)
  not in axis-space, but in raw -> MISSED AXIS (axes blind; raw residual = new-axis candidate)
with the caveat: a new cluster counts only if REPEATABLE across splits AND has
SUFFICIENT POINTS (else it's a distribution tail, not a defect).
"""
import numpy as np
from sklearn.metrics import silhouette_score


def separability(X, mask, thr=0.2):
    """silhouette of a planted binary split; >thr ~ separated."""
    if mask.sum() == 0 or (~mask).sum() == 0:
        return 0.0
    return silhouette_score(X, mask.astype(int))


def discriminate(sep_axis, sep_raw, thr=0.2):
    if sep_axis > thr:
        return "TUNING"
    if sep_raw > thr:
        return "MISSED AXIS"
    return "tail/noise"


def new_cluster_dual_sweep(axis_pts, raw_pts, frame_centers_axis, taus):
    """For a held-out set, sweep the new-cluster distance threshold in BOTH
    axis-space and raw-space. A sharp knee in axis-space but a smeared one in
    raw-space is the quantitative signature of a missed axis."""
    d_axis = np.min(np.linalg.norm(axis_pts[:, None] - frame_centers_axis[None], axis=2), axis=1)
    d_axis = d_axis / (np.median(d_axis) + 1e-9)
    frac_axis = np.array([np.mean(d_axis >= t) for t in taus])
    # raw-space self-distance (no frame centers in raw); use kNN-style spread proxy
    # here we report the curve of "fraction far from raw centroid"
    raw_c = raw_pts.mean(0)
    d_raw = np.linalg.norm(raw_pts - raw_c, axis=1)
    d_raw = d_raw / (np.median(d_raw) + 1e-9)
    frac_raw = np.array([np.mean(d_raw >= t) for t in taus])
    return frac_axis, frac_raw


def is_defect(verdicts, locations, sizes, min_points=50):
    """A new cluster is a DEFECT only if repeatable (same verdict across splits)
    AND backed by sufficient points. Sparse + location-varying = tail, not defect."""
    repeatable = len(set(verdicts)) == 1
    sufficient = max(sizes) >= min_points if sizes else False
    same_location = len(set(locations)) == 1
    return repeatable and sufficient and same_location, {
        "repeatable": repeatable, "sufficient": sufficient,
        "same_location": same_location, "verdict": verdicts[0] if repeatable else "mixed"}
