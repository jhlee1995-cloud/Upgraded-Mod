"""
frame/cache.py -- Stage B cache reader (provenance-aware).

Reads the caches Stage A's extractor wrote (via cache_audit), source-agnostic:
works identically on synthetic stand-in caches and real RunPod caches. Stage B
never inspects the source flag for logic; it's there only for the audit trail.

Cache kinds:
  clean_acts.npy        : raw clean activations (per image) -> coordinate frame
  point_<dataset>.npy   : single-batch axis vectors (per batch) -> disturbance points
  stream_<corr>_<order> : sequence axis vectors (per stream)  -> sequence-axis tests
"""
import json
import os
import numpy as np


class FrameCache:
    def __init__(self, cache_dir):
        self.dir = cache_dir
        self._meta_cache = {}

    def _load(self, name):
        npy = os.path.join(self.dir, f"{name}.npy")
        meta_path = os.path.join(self.dir, f"{name}.meta.json")
        arr = np.load(npy)
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self._meta_cache[name] = meta
        return arr, meta

    def clean_acts(self):
        """Raw clean activation cloud (N_images, feat_dim) for the coordinate frame."""
        arr, _ = self._load("clean_acts")
        return arr

    def point(self, dataset):
        """Single-batch axis vectors (N_batches, n_single_axes) for a dataset."""
        arr, _ = self._load(f"point_{dataset}")
        return arr

    def stream(self, corruption, order):
        """Sequence axis vectors (N_streams, n_seq_axes)."""
        arr, _ = self._load(f"stream_{corruption}_{order}")
        return arr

    def list_caches(self):
        return sorted(f[:-4] for f in os.listdir(self.dir) if f.endswith(".npy"))

    def is_synthetic(self):
        """True if any cache is flagged synthetic (the stand-in stage)."""
        for name in self.list_caches():
            mp = os.path.join(self.dir, f"{name}.meta.json")
            if os.path.exists(mp):
                if json.load(open(mp)).get("synthetic"):
                    return True
        return False
