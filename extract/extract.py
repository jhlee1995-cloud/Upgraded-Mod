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
                          build_stream_batches, build_ramp_stream,
                          build_recovery_stream, _cycle)


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


def build_cluster_structure_dump(args, hooks, ref, device, dataset_fps):
    """Does penult space even HAVE valleys? Dump the raw geometry to find out:
      - per-class centers + within-class spread + between-class distances (separation)
      - per-class covariance eigenvalues (sphere vs elongated -> mean_nearest vs mahalanobis)
      - 2D PCA projection of clean samples (+ type-b) for visual valley inspection
      - where misclassified (type-b) samples land relative to the valleys
    Saves npy arrays under <out>/structure/ for structure_geometry.py."""
    if not args.cluster_struct:
        return
    print("\n[cluster structure] probing penult geometry (do valleys exist?):")
    import torch.nn.functional as F
    tf = cifar_transform()
    loader, _ = make_loader(args.volume, "cifar10", tf, args.batch, args.download)

    feats_all, labels_all, preds_all, conf_all = [], [], [], []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        f, logits = hooks.forward(x)
        feats_all.append(f[args.layer].cpu().numpy())
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds_all.append(probs.argmax(1))
        conf_all.append(probs.max(1))
        labels_all.append(y.numpy())
        seen += len(y)
        if seen >= args.struct_samples:
            break
    F_ = np.vstack(feats_all)
    lab = np.concatenate(labels_all)
    pred = np.concatenate(preds_all)
    conf = np.concatenate(conf_all)
    print(f"  collected {len(F_)} samples ({(pred!=lab).sum()} misclassified)")

    # per-class centers, within spread, covariance eigenvalues
    nc = ref.n_classes
    centers = np.zeros((nc, F_.shape[1]))
    within = np.zeros(nc)
    eigs = np.zeros((nc, min(F_.shape[1], 10)))  # top-10 cov eigenvalues
    for c in range(nc):
        m = lab == c
        if m.sum() < 2:
            continue
        Xc = F_[m]
        centers[c] = Xc.mean(0)
        within[c] = np.linalg.norm(Xc - centers[c], axis=1).mean()
        cov = np.cov(Xc, rowvar=False)
        ev = np.linalg.eigvalsh(cov)[::-1][:10]
        eigs[c, :len(ev)] = ev

    # between-class center distances
    between = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(between, np.nan)

    # 2D PCA projection (clean) for visual inspection
    Fm = F_ - F_.mean(0)
    U, S, Vt = np.linalg.svd(Fm, full_matrices=False)
    proj2d = Fm @ Vt[:2].T  # (N, 2)

    # save under structure/
    sdir = os.path.join(args.out, "structure")
    os.makedirs(sdir, exist_ok=True)
    np.savez(os.path.join(sdir, "cluster_geometry.npz"),
             centers=centers, within=within, between=between, eigs=eigs,
             proj2d=proj2d, labels=lab, preds=pred, conf=conf,
             explained_var=(S**2 / (S**2).sum())[:10])
    print(f"  saved structure/cluster_geometry.npz")
    print(f"  within-class spread: mean {within.mean():.3f}")
    print(f"  between-class dist:  mean {np.nanmean(between):.3f}")
    sep = np.nanmean(between) / (within.mean() + 1e-9)
    print(f"  separation ratio (between/within): {sep:.2f}  "
          f"({'valleys exist' if sep > 1.5 else 'clusters overlap -- weak valleys'})")


def build_typeb_dump(args, hooks, ref, device, dataset_fps):
    """TRUE type-b: CIFAR-10 test images the model MISCLASSIFIES. Same distribution
    (energy-normal by construction) + wrong prediction = type-b by definition. Unlike
    iSUN (far-OOD, energy drops), these are in-distribution errors. Split into:
      diag_cifar10_correct  -- correctly classified (the 'clean' reference for type-b)
      diag_cifar10_wrong    -- all misclassified
      diag_cifar10_confwrong-- misclassified AND high-confidence (the hardest type-b)
    Test: does CLUSTER_DISTANCE fire on wrong-but-confident while CONSENSUS stays silent?"""
    if not args.typeb:
        return
    print("\n[type-b dump] CIFAR-10 misclassified samples (true energy-normal type-b):")
    import torch.nn.functional as F
    from axis_registry import valley_margin, valley_entropy
    tf = cifar_transform()
    loader, _ = make_loader(args.volume, "cifar10", tf, args.batch, args.download)
    diag_axes = dict(SINGLE_BATCH_AXES)
    diag_axes["VALLEY_MARGIN"] = valley_margin
    diag_axes["VALLEY_ENTROPY"] = valley_entropy
    axis_order = list(diag_axes)

    # collect per-sample: activation, predicted, true, confidence
    correct_acts, wrong_acts, confwrong_acts = [], [], []
    n_seen = 0
    for x, y in loader:
        x = x.to(device)
        feats, logits = hooks.forward(x)
        X = feats[args.layer].cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(1)
        conf = probs.max(1)
        y = y.numpy()
        for i in range(len(y)):
            row = X[i:i+1]  # single-sample "batch" (1, C)
            if pred[i] == y[i]:
                correct_acts.append(row)
            else:
                wrong_acts.append(row)
                if conf[i] > args.conf_thresh:
                    confwrong_acts.append(row)
        n_seen += len(y)
        if n_seen >= args.typeb_samples:
            break

    print(f"  scanned {n_seen} samples: {len(correct_acts)} correct, "
          f"{len(wrong_acts)} wrong, {len(confwrong_acts)} conf-wrong (>{args.conf_thresh})")

    # axes are batch-level; group single samples into pseudo-batches of size args.batch
    def to_batches(acts, name, label):
        if len(acts) < args.batch:
            print(f"  {name}: only {len(acts)} samples (<{args.batch}); skipping")
            return
        A = np.vstack(acts)
        nb = len(A) // args.batch
        vectors = []
        for b in range(nb):
            Xb = A[b*args.batch:(b+1)*args.batch]
            vectors.append([diag_axes[a](Xb, ref).mean() for a in axis_order])
        arr = np.array(vectors)
        save_cache(arr, name, args.out, meta=dict(
            source="real", backbone=args.arch, layer=args.layer, dataset=label,
            axis_formulas={"columns": axis_order},
            notes=f"type-b dump: {label} ({len(A)} samples -> {nb} pseudo-batches)"))
        print(f"  {name}: {arr.shape} -> saved")

    to_batches(correct_acts, "diag_cifar10_correct", "cifar10-correct")
    to_batches(wrong_acts, "diag_cifar10_wrong", "cifar10-wrong (type-b)")
    to_batches(confwrong_acts, "diag_cifar10_confwrong", "cifar10-confwrong (hard type-b)")


def build_diagnostic_dump(args, hooks, ref, device, dataset_fps):
    """Per-batch point-axis vectors for CLEAN and each corruption SEPARATELY.
    Enables within-group correlation (clean-only, corruption-only) which removes
    the clean-corrupt common mode that inflates correlations in mixed streams.
    This is the test for whether the structure axes are truly redundant or co-quiet."""
    if not args.diag:
        return
    print("\n[diagnostic dump] per-batch axes + valley geometry, clean + per-corruption:")
    tf = cifar_transform()
    # 4 standard axes + 2 valley-geometry candidates (margin, entropy)
    from axis_registry import valley_margin, valley_entropy
    diag_axes = dict(SINGLE_BATCH_AXES)
    diag_axes["VALLEY_MARGIN"] = valley_margin
    diag_axes["VALLEY_ENTROPY"] = valley_entropy
    axis_order = list(diag_axes)

    def dump(loader, name, ds_label):
        vectors = []
        seen = 0
        for x, y in loader:
            feats, _ = hooks.forward(x.to(device))
            X = feats[args.layer].cpu().numpy()
            vectors.append([diag_axes[a](X, ref).mean() for a in axis_order])
            seen += 1
            if seen >= args.diag_batches:
                break
        arr = np.array(vectors)
        save_cache(arr, name, args.out, meta=dict(
            source="real", backbone=args.arch, layer=args.layer, dataset=ds_label,
            axis_formulas={"columns": axis_order},
            notes=f"per-batch axes + valley geom for {ds_label} (diagnostic)"))
        print(f"  {name}: {arr.shape} -> saved")

    # clean
    clean_loader, _ = make_loader(args.volume, "cifar10", tf, args.batch, args.download)
    dump(clean_loader, "diag_clean", "cifar10")
    # each corruption at the chosen severity
    if args.cifar10c:
        for corruption in args.stream_corruptions:
            ld, _ = make_loader(args.volume, "cifar10c", tf, args.batch, args.download,
                                corruption=corruption, severity=args.severity)
            dump(ld, f"diag_{corruption}", f"cifar10c:{corruption}@sev{args.severity}")
        # severity sweep: dump severities 1,2,3 to escape the AUC ceiling
        if args.sev_sweep:
            print("  [severity sweep] dumping severities 1,2,3 (escape AUC ceiling):")
            for corruption in args.stream_corruptions:
                for sev in (1, 2, 3):
                    ld, _ = make_loader(args.volume, "cifar10c", tf, args.batch,
                                        args.download, corruption=corruption, severity=sev)
                    dump(ld, f"diag_{corruption}_s{sev}", f"cifar10c:{corruption}@sev{sev}")
    # near-OOD too if present
    for ds in ["cifar100", "isun"]:
        try:
            ld, _ = make_loader(args.volume, ds, tf, args.batch, args.download)
            dump(ld, f"diag_{ds}", ds)
        except Exception as e:
            print(f"  diag_{ds} skipped: {repr(e)[:60]}")


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
                # ALSO compute the 4 point-axis means over the stream's batches, so
                # point + sequence axes share a sampling unit (per stream) -> enables
                # a true joint 7-axis covariance (exp0). Stored as extra columns.
                point_means = [np.mean([SINGLE_BATCH_AXES[a](h, ref).mean() for h in acts])
                               for a in SINGLE_BATCH_AXES]
                row = point_means + [seq[c] for c in SEQUENCE_AXIS_COLUMNS]
                stream_vectors.append(row)
            arr = np.array(stream_vectors)  # (n_streams, 4 + 5)
            joint_cols = list(SINGLE_BATCH_AXES) + SEQUENCE_AXIS_COLUMNS
            save_cache(arr, f"stream_{corruption}_{order}", args.out, meta=dict(
                source="real", backbone=args.arch, layer=args.layer,
                dataset=f"cifar10c:{corruption}@sev{args.severity}",
                dataset_fp=dataset_fps.get("cifar10c"),
                axis_formulas={"columns": joint_cols,
                               "persist_mode": args.persist_mode,
                               "note": "point-axis means + signed/abs sequence axes; "
                                       "9 cols on one sampling unit -> joint 7-axis exp0"},
                notes=f"joint axes (9 cols); order={order}, ratio={args.ratio}, "
                      f"stream_len={args.stream_len}, persist_mode={args.persist_mode}"))
            print(f"  stream_{corruption}_{order}: {arr.shape} -> saved (4 point + 5 seq)")

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
                    point_means = [np.mean([SINGLE_BATCH_AXES[a](h, ref).mean() for h in acts])
                                   for a in SINGLE_BATCH_AXES]
                    stream_vectors.append(point_means + [seq[c] for c in SEQUENCE_AXIS_COLUMNS])
                arr = np.array(stream_vectors)
                joint_cols = list(SINGLE_BATCH_AXES) + SEQUENCE_AXIS_COLUMNS
                save_cache(arr, f"ramp_{corruption}_{schedule}", args.out, meta=dict(
                    source="real", backbone=args.arch, layer=args.layer,
                    dataset=f"cifar10c:{corruption}@ramp1-5",
                    dataset_fp=dataset_fps.get("cifar10c"),
                    axis_formulas={"columns": joint_cols,
                                   "persist_mode": args.persist_mode,
                                   "schedule": schedule},
                    notes=f"joint axes (9 cols) RAMP {schedule}; gradual drift; "
                          f"stream_len={args.stream_len}"))
                print(f"  ramp_{corruption}_{schedule}: {arr.shape} -> saved (4 point + 5 seq)")

    # --- RECOVERY streams: severity up then back down (map row 9, adapt-stop) ---
    if args.recovery:
        print("\n  [recovery streams] severity up->peak->down to clean "
              "(does any axis read DIRECTION?):")
        clean_loader, _ = make_loader(args.volume, "cifar10", tf, args.batch, args.download)
        for corruption in args.stream_corruptions:
            sev_loaders = {}
            for sev in range(1, 6):
                ld, _ = make_loader(args.volume, "cifar10c", tf, args.batch,
                                    args.download, corruption=corruption, severity=sev)
                sev_loaders[sev] = ld
            stream_vectors = []
            for s in range(args.n_streams):
                plan = build_recovery_stream(sev_loaders, clean_loader, args.stream_len)
                acts = []
                for x, _sev in plan:
                    feats, _ = hooks.forward(x.to(device))
                    acts.append(feats[args.layer].cpu().numpy())
                seq = compute_sequence_axes(acts, ref, mode=args.persist_mode)
                point_means = [np.mean([SINGLE_BATCH_AXES[a](h, ref).mean() for h in acts])
                               for a in SINGLE_BATCH_AXES]
                stream_vectors.append(point_means + [seq[c] for c in SEQUENCE_AXIS_COLUMNS])
            arr = np.array(stream_vectors)
            joint_cols = list(SINGLE_BATCH_AXES) + SEQUENCE_AXIS_COLUMNS
            save_cache(arr, f"recovery_{corruption}", args.out, meta=dict(
                source="real", backbone=args.arch, layer=args.layer,
                dataset=f"cifar10c:{corruption}@recovery",
                dataset_fp=dataset_fps.get("cifar10c"),
                axis_formulas={"columns": joint_cols, "persist_mode": args.persist_mode},
                notes=f"joint axes (9 cols) RECOVERY up-down; row9; stream_len={args.stream_len}"))
            print(f"  recovery_{corruption}: {arr.shape} -> saved (4 point + 5 seq)")


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

    # 2b) diagnostic per-batch dump (clean + per-corruption) for common-mode test
    build_diagnostic_dump(args, hooks, ref, device, dataset_fps)

    # 2c) TRUE type-b dump (CIFAR-10 misclassified = energy-normal confident-wrong)
    build_typeb_dump(args, hooks, ref, device, dataset_fps)

    # 2d) cluster-structure probe (do valleys exist? where is type-b?)
    build_cluster_structure_dump(args, hooks, ref, device, dataset_fps)

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
    ap.add_argument("--diag", action="store_true",
                    help="dump per-batch axes for clean + each corruption (common-mode test)")
    ap.add_argument("--diag-batches", type=int, default=60, help="batches per diagnostic dump")
    ap.add_argument("--typeb", action="store_true",
                    help="dump CIFAR-10 misclassified samples (true energy-normal type-b)")
    ap.add_argument("--typeb-samples", type=int, default=10000, help="CIFAR-10 samples to scan for type-b")
    ap.add_argument("--conf-thresh", type=float, default=0.7, help="confidence threshold for conf-wrong type-b")
    ap.add_argument("--sev-sweep", action="store_true",
                    help="dump diag at severities 1,2,3 to escape the AUC ceiling")
    ap.add_argument("--cluster-struct", action="store_true",
                    help="probe penult cluster geometry (do valleys exist? where is type-b?)")
    ap.add_argument("--recovery", action="store_true",
                    help="recovery streams (severity up then down; map row 9, adapt-stop)")
    ap.add_argument("--struct-samples", type=int, default=10000, help="samples for cluster-structure probe")
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
