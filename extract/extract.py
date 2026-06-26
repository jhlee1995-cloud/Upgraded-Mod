"""
STAGE A -- extract.py  [runs on RunPod GPU]

Ties the three systems together:
  populate_data.resolve_path  -> find datasets via manifest (mount-agnostic)
  axis_registry               -> the 7 chosen axis formulas
  cache_audit.save_cache      -> write 6/7-axis vectors WITH provenance

Produces two cache kinds (Stage B consumes both):
  POINT  cache: single-batch axes, one 6D-ish point per batch  -> coordinate frame
  STREAM cache: sequence axes, one scalar per stream            -> sequence-axis tests

After layer_sweep decides the layer(s), pass --layer; extract pulls activations
from that layer for every dataset and writes the caches.

Usage on pod:
  python extract.py --volume /runpod-volume --layer penult \
                    --out /runpod-volume/cache/run1 --arch cifar10_resnet20
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbone import load_backbone, MultiLayerHooks
from axis_registry import (AxisRef, SINGLE_BATCH_AXES, SEQUENCE_AXES,
                           compute_sequence_axes, SEQUENCE_AXIS_COLUMNS)
from cache_audit import save_cache
from populate_data import resolve_path, sha256_of_dir, DATASETS_SUBDIR
from data_loaders import (CIFAR10C, ImageFolderFlat, CIFAR10C_CORRUPTIONS,
                          build_stream_batches, build_ramp_stream, _cycle)


AXIS_FORMULAS = {
    "DEVIATION": "l2_signed", "CONSENSUS": "disagree_rate",
    "CLUSTER_DISTANCE": "mean_nearest", "SUBNET_CONSENSUS": "mean_entropy",
    "DRIFT_COH": "window_cosine", "PERSIST": "streak", "CLUST_DRIFT": "consec_cosine",
}


def cifar_transform(img_size=32):
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    return T.Compose([T.Resize(img_size), T.ToTensor(), norm])


def make_loader(volume, dataset, transform, batch, download, corruption=None, severity=3):
    """Resolve dataset path via manifest; build a loader.
    dataset in {cifar10, cifar100, svhn, cifar10c, isun}."""
    try:
        root = resolve_path(volume, dataset)
    except (KeyError, FileNotFoundError):
        root = os.path.join(volume, DATASETS_SUBDIR, dataset)
    if dataset == "cifar10c":
        ds = CIFAR10C(root, corruption, severity, transform=transform)
    elif dataset == "isun":
        ds = ImageFolderFlat(root, transform=transform)
    else:
        builders = {
            "cifar10":  lambda: torchvision.datasets.CIFAR10(root, train=False, download=download, transform=transform),
            "cifar100": lambda: torchvision.datasets.CIFAR100(root, train=False, download=download, transform=transform),
            "svhn":     lambda: torchvision.datasets.SVHN(root, split="test", download=download, transform=transform),
        }
        if dataset not in builders:
            raise ValueError(f"dataset {dataset} loader not wired")
        ds = builders[dataset]()
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                       num_workers=2, drop_last=True), root


def extract_layer_acts(hooks, loader, layer, device, n_batches, keep_labels=True):
    """Collect (N, C) activations from ONE layer, plus labels."""
    acts, labels = [], []
    seen = 0
    for x, y in loader:
        feats, _ = hooks.forward(x.to(device))
        acts.append(feats[layer].cpu().numpy())
        labels.append(np.asarray(y))
        seen += 1
        if seen >= n_batches:
            break
    return np.concatenate(acts), np.concatenate(labels)


def build_point_cache(args, hooks, ref, datasets, device, dataset_fps):
    """For each dataset: single-batch axes -> (N_batches, 4) point vectors.
    One vector per BATCH (axes are batch-level statistics)."""
    tf = cifar_transform()
    axis_order = list(SINGLE_BATCH_AXES)
    for ds in datasets:
        loader, root = make_loader(args.volume, ds, tf, args.batch, args.download)
        # collect per-batch axis vectors
        vectors = []
        seen = 0
        for x, y in loader:
            feats, _ = hooks.forward(x.to(device))
            X = feats[args.layer].cpu().numpy()
            vec = [SINGLE_BATCH_AXES[a](X, ref).mean() for a in axis_order]  # batch-level scalar per axis
            vectors.append(vec)
            seen += 1
            if seen >= args.test_batches:
                break
        arr = np.array(vectors)  # (N_batches, 4)
        save_cache(arr, f"point_{ds}", args.out, meta=dict(
            source="real", backbone=args.arch, layer=args.layer, dataset=ds,
            dataset_fp=dataset_fps.get(ds),
            axis_formulas={a: AXIS_FORMULAS[a] for a in axis_order},
            notes="single-batch axes, one row per batch (batch-level mean)"))
        print(f"  point_{ds}: {arr.shape} -> saved")


def build_clean_frame_ref(args, hooks, device):
    """Build the AxisRef (per-subnet centers) from clean CIFAR-10."""
    tf = cifar_transform()
    loader, root = make_loader(args.volume, "cifar10", tf, args.batch, args.download)
    acts, labels = extract_layer_acts(hooks, loader, args.layer, device, args.calib_batches)
    ref = AxisRef(acts, labels, n_classes=args.n_classes)
    print(f"frame ref built from {len(acts)} clean acts at layer '{args.layer}' "
          f"(feat_dim {acts.shape[1]})")
    # also save the raw clean point cloud for Stage B coordinate frame
    return ref, acts, labels


def build_stream_cache(args, hooks, ref, device, dataset_fps):
    """Sequence axes need STREAMS (real-time movement). Build streams mixing clean
    CIFAR-10 with a CIFAR-10-C corruption, in both shuffle and block order, run them
    through the hook, compute the 3 sequence-axis scalars per stream."""
    if not args.cifar10c:
        print("\n[stream cache] --cifar10c not enabled; skipping sequence-axis extraction")
        print("  (sequence axes DRIFT_COH/PERSIST/CLUST_DRIFT need CIFAR-10-C streams)")
        return
    tf = cifar_transform()
    clean_loader, _ = make_loader(args.volume, "cifar10", tf, args.batch, args.download)

    for corruption in args.stream_corruptions:
        corr_loader, _ = make_loader(args.volume, "cifar10c", tf, args.batch,
                                     args.download, corruption=corruption,
                                     severity=args.severity)
        for order in ("shuffle", "block"):
            stream_vectors = []
            for s in range(args.n_streams):
                rng = np.random.default_rng(1000 + s)
                plan = build_stream_batches(clean_loader, corr_loader, args.ratio,
                                            order, args.stream_len, args.batch, rng)
                # run each batch through hook -> list of (B, C) activations
                acts = []
                for x, _is_corrupt in plan:
                    feats, _ = hooks.forward(x.to(device))
                    acts.append(feats[args.layer].cpu().numpy())
                # 5 sequence columns: signed+abs for both drifts, plus PERSIST
                seq = compute_sequence_axes(acts, ref, mode=args.persist_mode)
                stream_vectors.append([seq[c] for c in SEQUENCE_AXIS_COLUMNS])
            arr = np.array(stream_vectors)  # (n_streams, 5)
            save_cache(arr, f"stream_{corruption}_{order}", args.out, meta=dict(
                source="real", backbone=args.arch, layer=args.layer,
                dataset=f"cifar10c:{corruption}@sev{args.severity}",
                dataset_fp=dataset_fps.get("cifar10c"),
                axis_formulas={"columns": SEQUENCE_AXIS_COLUMNS,
                               "persist_mode": args.persist_mode,
                               "note": "signed+abs drift forms stored; sign TBD from data"},
                notes=f"sequence axes (5 cols); order={order}, ratio={args.ratio}, "
                      f"stream_len={args.stream_len}, persist_mode={args.persist_mode}"))
            print(f"  stream_{corruption}_{order}: {arr.shape} -> saved "
                  f"[{', '.join(SEQUENCE_AXIS_COLUMNS)}]")

    # --- RAMP-SEVERITY streams: gradual drift (the proper DRIFT_COH/CLUST_DRIFT test) ---
    if args.ramp:
        print("\n  [ramp streams] gradual severity 1->5 drift "
              "(proper stimulus for DRIFT_COH/CLUST_DRIFT):")
        for corruption in args.stream_corruptions:
            # one loader per severity for this corruption
            sev_loaders = {}
            for sev in range(1, 6):
                ld, _ = make_loader(args.volume, "cifar10c", tf, args.batch,
                                    args.download, corruption=corruption, severity=sev)
                sev_loaders[sev] = ld
            for schedule in ("linear", "late"):
                stream_vectors = []
                for s in range(args.n_streams):
                    plan = build_ramp_stream(sev_loaders, args.stream_len, schedule=schedule)
                    acts = []
                    for x, _sev in plan:
                        feats, _ = hooks.forward(x.to(device))
                        acts.append(feats[args.layer].cpu().numpy())
                    seq = compute_sequence_axes(acts, ref, mode=args.persist_mode)
                    stream_vectors.append([seq[c] for c in SEQUENCE_AXIS_COLUMNS])
                arr = np.array(stream_vectors)
                save_cache(arr, f"ramp_{corruption}_{schedule}", args.out, meta=dict(
                    source="real", backbone=args.arch, layer=args.layer,
                    dataset=f"cifar10c:{corruption}@ramp1-5",
                    dataset_fp=dataset_fps.get("cifar10c"),
                    axis_formulas={"columns": SEQUENCE_AXIS_COLUMNS,
                                   "persist_mode": args.persist_mode,
                                   "schedule": schedule},
                    notes=f"RAMP severity 1->5 ({schedule}); gradual drift test for "
                          f"DRIFT_COH/CLUST_DRIFT; stream_len={args.stream_len}"))
                print(f"  ramp_{corruption}_{schedule}: {arr.shape} -> saved")


def main(args):
    device = args.device
    os.makedirs(args.out, exist_ok=True)
    model, src = load_backbone(args.arch, device)
    print("MODEL:", src, "| layer:", args.layer)
    hooks = MultiLayerHooks(model)
    if args.layer not in hooks.layer_names:
        raise ValueError(f"layer '{args.layer}' not hooked; available: {hooks.layer_names}")

    # dataset fingerprints (provenance)
    dataset_fps = {}
    for ds in ["cifar10", "cifar100", "svhn", "cifar10c", "isun"]:
        try:
            root = resolve_path(args.volume, ds)
            fp, _ = sha256_of_dir(root)
            dataset_fps[ds] = fp
        except Exception:
            dataset_fps[ds] = None

    # 1) clean frame reference
    ref, clean_acts, clean_lab = build_clean_frame_ref(args, hooks, device)
    # save clean point cloud (per-image, for the coordinate frame in Stage B)
    save_cache(clean_acts, "clean_acts", args.out, meta=dict(
        source="real", backbone=args.arch, layer=args.layer, dataset="cifar10",
        dataset_fp=dataset_fps.get("cifar10"),
        notes="raw clean activations (per image) for coordinate-frame building"))
    print(f"  clean_acts: {clean_acts.shape} -> saved")

    # 2) point caches for disturbance/near-OOD datasets
    print("\nbuilding POINT caches (single-batch axes):")
    point_datasets = ["cifar100", "svhn"]
    if args.isun:
        point_datasets.append("isun")
    build_point_cache(args, hooks, ref, point_datasets, device, dataset_fps)

    # 3) stream caches for sequence axes (need CIFAR-10-C)
    build_stream_cache(args, hooks, ref, device, dataset_fps)

    print(f"\nStage A extraction done -> {args.out}")
    print("run: python cache_audit.py --dir", args.out, "--audit --verify")
    hooks.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True, help="volume mount root on this pod")
    ap.add_argument("--out", required=True, help="cache output dir (e.g. /runpod-volume/cache/run1)")
    ap.add_argument("--layer", default="penult", help="layer to extract from (from layer_sweep)")
    ap.add_argument("--arch", default="cifar10_resnet20")
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--calib-batches", type=int, default=40)
    ap.add_argument("--test-batches", type=int, default=40)
    ap.add_argument("--isun", action="store_true", help="include iSUN near-OOD in point cache")
    ap.add_argument("--cifar10c", action="store_true", help="extract sequence axes from CIFAR-10-C streams")
    ap.add_argument("--ramp", action="store_true",
                    help="also extract ramp-severity streams (gradual drift; proper DRIFT_COH/CLUST_DRIFT test)")
    ap.add_argument("--stream-corruptions", nargs="*",
                    default=["fog", "gaussian_noise", "motion_blur"],
                    help="which CIFAR-10-C corruptions to stream")
    ap.add_argument("--severity", type=int, default=3, help="CIFAR-10-C severity for streams")
    ap.add_argument("--persist-mode", default="distance",
                    choices=["distance", "structure", "energy"],
                    help="aggregate-anomaly basis for PERSIST (distance validated; energy=legacy/blind)")
    ap.add_argument("--ratio", type=float, default=0.3, help="corruption fraction in streams")
    ap.add_argument("--n-streams", type=int, default=40, help="streams per (corruption,order)")
    ap.add_argument("--stream-len", type=int, default=8, help="batches per stream (T)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    main(args)
