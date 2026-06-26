"""
STAGE A -- layer sweep  [runs on RunPod GPU]

Question (approach c): is penultimate the right layer for all axes, or does each
axis peak at a different layer? Extracts the 4 single-batch axes from EVERY hooked
layer on clean vs a disturbance set, reports per-axis AUC per layer.

  all axes peak at penult        -> single-layer (penult); simple
  axes peak at different layers   -> multi-layer; use per-axis best layer

Sequence axes (DRIFT_COH/PERSIST/CLUST_DRIFT) need streams -> deferred to the
stream-extraction step once real streams are built.

Data are real: CIFAR-10 (clean) vs a disturbance loader (CIFAR-10-C / CIFAR-100 /
iSUN / SVHN). Mount path via --data-root. Minimal deps.
"""
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backbone import load_backbone, MultiLayerHooks
from axis_registry import AxisRef, SINGLE_BATCH_AXES


def cifar_transform(img_size=32):
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    return T.Compose([T.Resize(img_size), T.ToTensor(), norm])


def collect_layer_acts(hooks, loader, device, max_batches, with_labels=True):
    """Run loader through hooks; return {layer: (N, C)} and labels (N,)."""
    per_layer = {name: [] for name in hooks.layer_names}
    labels = []
    seen = 0
    for batch in loader:
        x, y = batch
        feats, _ = hooks.forward(x.to(device))
        for name in hooks.layer_names:
            per_layer[name].append(feats[name].cpu().numpy())
        labels.append(y.numpy())
        seen += 1
        if seen >= max_batches:
            break
    per_layer = {k: np.concatenate(v) for k, v in per_layer.items()}
    labels = np.concatenate(labels)
    return per_layer, labels


def run_layer_sweep(args):
    device = args.device
    model, src = load_backbone(args.arch, device)
    print("MODEL:", src)
    hooks = MultiLayerHooks(model)
    print("hooked layers:", hooks.layer_names)

    tf = cifar_transform()
    # clean = CIFAR-10 test
    clean_set = torchvision.datasets.CIFAR10(args.data_root, train=False,
                                             download=args.download, transform=tf)
    clean_loader = torch.utils.data.DataLoader(clean_set, batch_size=args.batch,
                                               shuffle=True, num_workers=2, drop_last=True)
    # disturbance = CIFAR-100 (near-OOD, type-b-bearing) by default
    if args.disturbance == "cifar100":
        dist_set = torchvision.datasets.CIFAR100(args.data_root, train=False,
                                                 download=args.download, transform=tf)
    elif args.disturbance == "svhn":
        dist_set = torchvision.datasets.SVHN(args.data_root, split="test",
                                             download=args.download, transform=tf)
    else:
        raise ValueError(f"disturbance {args.disturbance} not wired (add CIFAR-10-C/iSUN loader)")
    dist_loader = torch.utils.data.DataLoader(dist_set, batch_size=args.batch,
                                              shuffle=True, num_workers=2, drop_last=True)

    print(f"\ncollecting clean activations ({args.calib_batches} batches)...")
    clean_acts, clean_lab = collect_layer_acts(hooks, clean_loader, device, args.calib_batches)
    print(f"collecting disturbance ({args.disturbance}) activations...")
    dist_acts, _ = collect_layer_acts(hooks, dist_loader, device, args.test_batches)

    # split clean: half to fit refs (centers), half as clean test
    print("\n" + "=" * 64)
    print(f"LAYER SWEEP -- per-axis AUC (clean vs {args.disturbance})")
    print("=" * 64)
    layers = hooks.layer_names
    header = f"{'axis':18s} | " + " ".join(f"{ln:>9s}" for ln in layers)
    print(header)
    print("-" * len(header))

    results = {}
    for axis_name, fn in SINGLE_BATCH_AXES.items():
        aucs = []
        for ln in layers:
            Xc = clean_acts[ln]
            Xd = dist_acts[ln]
            # fit ref on first half of clean, score on second half + disturbance
            half = len(Xc) // 2
            ref = AxisRef(Xc[:half], clean_lab[:half], n_classes=args.n_classes)
            s_clean = fn(Xc[half:], ref)
            s_dist = fn(Xd, ref)
            y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_dist))]
            s = np.r_[s_clean, s_dist]
            auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
            aucs.append(auc)
        results[axis_name] = dict(zip(layers, aucs))
        best_layer = layers[int(np.argmax(aucs))]
        print(f"{axis_name:18s} | " + " ".join(f"{a:9.3f}" for a in aucs)
              + f"   best={best_layer}")

    # verdict
    print("\n" + "=" * 64)
    best_layers = {a: max(r, key=r.get) for a, r in results.items()}
    unique_best = set(best_layers.values())
    print(f"per-axis best layer: {best_layers}")
    if len(unique_best) == 1:
        print(f"VERDICT: all axes peak at '{list(unique_best)[0]}' -> SINGLE-LAYER")
    else:
        print(f"VERDICT: axes peak at different layers {unique_best} -> MULTI-LAYER worth it")
    hooks.close()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data",
                    help="dataset mount path (match the volume mount, e.g. /runpod-volume/data)")
    ap.add_argument("--arch", default="cifar10_resnet20")
    ap.add_argument("--disturbance", default="cifar100", choices=["cifar100", "svhn"])
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--calib-batches", type=int, default=40)
    ap.add_argument("--test-batches", type=int, default=40)
    ap.add_argument("--download", action="store_true",
                    help="allow torchvision auto-download (omit if data already on volume)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run_layer_sweep(args)
