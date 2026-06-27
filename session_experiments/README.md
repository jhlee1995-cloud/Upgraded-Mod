# Session Experiments — Valleys, type-b, Scale, Blind-Spots, Three Gates

Standalone experiment scripts from the major real-data session (CIFAR → ImageNet). Each is
self-contained: it loads a backbone, runs an experiment, prints results. Run from the repo root
with `python session_experiments/<script>.py --volume /workspace`.

These complement the in-repo analysis scripts (`analysis/`) that read cached extraction output.
The scripts here re-extract / re-run as needed and were the actual instruments behind the
findings in `MASTER_SUMMARY.md`.

## Catalog

| script | what it tested | key result |
|--------|----------------|------------|
| `mahalanobis_typeb.py` | Mahalanobis vs Euclidean cluster-distance + type-b at batch 32 | Mahalanobis helps fog/contrast (+0.11–0.14), not elsewhere; type-b all axes 1.00 at batch (ceiling) |
| `sample_and_scale.py` | per-sample type-b (no batch averaging) + ResNet20 vs ResNet56 | margin best per-sample (0.895); holds/strengthens at scale (0.905) — but accuracy-confounded |
| `uncovered_errors.py` | adversarial / partial-corruption / distribution-shift, each swept over a range | adversarial + partial caught; **distribution shift flat ~0.5 = structural blind spot** |
| `density_accel.py` | density-acceleration mechanism (stable/gradual/sudden streams, CIFAR) | sudden peak accel 3.4× gradual/stable; velocity alone fails → **acceleration necessary** |
| `imagenet_extract.py` | ResNet50 penult (2048d) extraction + valley separation + per-sample type-b | valley sep 1.14 (shallow); margin 0.800 (best, rank preserved) |
| `imagenet_valley_check.py` | is the shallow ImageNet valley a sample-count artifact? (min-samples sweep) | flat 1.13–1.14 across 3→30 samples/class → **genuinely shallow, not artifact** |
| `imagenet_density.py` | density-acceleration on SEMANTIC classes (ambulance/fire-engine vs cat/mug) | all spikes 5–6× stable; critical ≈ mundane → **class-agnostic gate + separate policy layer** |

## Related (in `analysis/`, read cached output)

- `squeeze_more.py` — signed-drift recovery (sep 1.99, replaces TRAJ_LOOP), valley measures
  across corruptions, per-axis detection floors.
- `fill_map.py` — fills temporal map rows (directional drift / non-directional / sustained /
  recovery) from stream caches.

## Notes

- CIFAR backbone: `chenyaofo/pytorch-cifar-models` (resnet20/56). ImageNet: torchvision
  `resnet50` IMAGENET1K_V2.
- ImageNet data: non-gated HF mirror `benjamin-paine/imagenet-1k-256x256` (parquet), downloaded
  via `huggingface_hub` directly — **not** the `datasets` library (it upgrades pandas/pyarrow
  and crashes the torch env).
- Caches: `/workspace/cache/run11` (CIFAR merged), `/workspace/cache/imagenet/penult_N.npz`.
