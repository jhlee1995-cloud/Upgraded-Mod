"""
data_loaders.py -- custom loaders torchvision lacks: CIFAR-10-C and iSUN,
plus a STREAM builder for sequence-axis extraction.

CIFAR-10-C layout (standard zenodo release): a directory of .npy files, one per
corruption (gaussian_noise.npy, fog.npy, ...), each shape (50000, 32, 32, 3) =
5 severities x 10000 images stacked (severity s = rows [s*10000:(s+1)*10000]).
labels.npy holds the shared labels (10000,), tiled across severities.

iSUN: a flat folder of images -> ImageFolder-style or raw image list.

Stream builder: arranges batches into TIME BLOCKS (sustained corruption) vs
SHUFFLE, for the sequence axes (DRIFT_COH/PERSIST/CLUST_DRIFT) which need
real-time movement.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


CIFAR10C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
SEVERITY_SIZE = 10000  # images per severity in the standard release


class CIFAR10C(Dataset):
    """One corruption at one severity from the CIFAR-10-C .npy release.

    root: directory containing <corruption>.npy and labels.npy
    corruption: name (e.g. 'fog'); severity: 1..5
    """
    def __init__(self, root, corruption, severity, transform=None):
        assert corruption in CIFAR10C_CORRUPTIONS, f"unknown corruption {corruption}"
        assert 1 <= severity <= 5
        self.transform = transform
        arr = np.load(os.path.join(root, f"{corruption}.npy"))     # (50000,32,32,3) uint8
        lab = np.load(os.path.join(root, "labels.npy"))            # (50000,) or (10000,)
        s = severity - 1
        self.data = arr[s * SEVERITY_SIZE:(s + 1) * SEVERITY_SIZE]
        if len(lab) == 50000:
            self.labels = lab[s * SEVERITY_SIZE:(s + 1) * SEVERITY_SIZE]
        else:
            self.labels = lab  # shared 10000 labels
        self.corruption = corruption
        self.severity = severity

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])


class ImageFolderFlat(Dataset):
    """iSUN / any flat image folder (no class subdirs). Label is a dummy 0."""
    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root, transform=None, max_images=None):
        self.transform = transform
        self.paths = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(self.EXTS):
                    self.paths.append(os.path.join(dirpath, f))
        self.paths.sort()
        if max_images:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise RuntimeError(f"no images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ---------------------------------------------------------------------------
# STREAM builder for sequence axes
# ---------------------------------------------------------------------------
def build_stream_batches(clean_loader, corruption_loader, ratio, order,
                         n_steps, batch, rng):
    """Yield a stream of n_steps batches mixing clean + corruption at `ratio`.
    order='shuffle' (i.i.d.) or 'block' (sustained corruption runs).
    Each yielded item is a batch tensor (B, 3, H, W) ready for the hook.

    Returns a LIST of (tensor, is_corrupt) so the extractor can run the hook.
    """
    clean_it = _cycle(clean_loader)
    corr_it = _cycle(corruption_loader)

    if order == "shuffle":
        plan = (rng.random(n_steps) < ratio)
    elif order == "block":
        # alternating clean-block / corrupt-block sized to honor ratio
        plan = np.zeros(n_steps, dtype=bool)
        block = max(1, n_steps // 6)
        t = 0
        toggle = False
        while t < n_steps:
            length = block if not toggle else max(1, int(block * ratio / max(1e-9, 1 - ratio)))
            plan[t:t + length] = toggle
            t += length
            toggle = not toggle
    else:
        raise ValueError(order)

    stream = []
    for is_corrupt in plan:
        x, _ = next(corr_it if is_corrupt else clean_it)
        stream.append((x, bool(is_corrupt)))
    return stream


def _cycle(loader):
    while True:
        for b in loader:
            yield b


if __name__ == "__main__":
    import argparse
    import torchvision.transforms as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar10c-root", help="dir with CIFAR-10-C .npy files")
    ap.add_argument("--isun-root", help="dir with iSUN images")
    args = ap.parse_args()

    tf = T.Compose([T.Resize(32), T.ToTensor()])

    if args.cifar10c_root:
        print("CIFAR-10-C check:")
        try:
            ds = CIFAR10C(args.cifar10c_root, "fog", 3, transform=tf)
            print(f"  fog@sev3: {len(ds)} images; sample shape {ds[0][0].shape}")
            print(f"  available corruptions: {len(CIFAR10C_CORRUPTIONS)}")
        except Exception as e:
            print(f"  FAILED: {repr(e)[:120]}")
    if args.isun_root:
        print("iSUN check:")
        try:
            ds = ImageFolderFlat(args.isun_root, transform=tf, max_images=100)
            print(f"  iSUN: {len(ds)} images; sample shape {ds[0][0].shape}")
        except Exception as e:
            print(f"  FAILED: {repr(e)[:120]}")
    if not (args.cifar10c_root or args.isun_root):
        ap.print_help()
