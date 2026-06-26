#!/usr/bin/env bash
# RunPod end-to-end: populate -> layer sweep -> extract -> audit -> exp0
# Usage: ./run_pod.sh /runpod-volume
set -euo pipefail

VOLUME="${1:?usage: ./run_pod.sh <volume-mount> (e.g. /runpod-volume)}"
RUN="${2:-run1}"
OUT="$VOLUME/cache/$RUN"

echo "=== install ==="
pip install -q -r requirements.txt

echo "=== 1. populate datasets (idempotent) ==="
python -m extract.populate_data --volume "$VOLUME" || true
python -m extract.populate_data --volume "$VOLUME" --verify

echo "=== 2. layer sweep ==="
python -m extract.layer_sweep --data-root "$VOLUME/datasets/cifar10" --download \
  || echo "(layer sweep needs CIFAR present; continuing)"

echo "=== 3. extract 7 axes -> $OUT ==="
python -m extract.extract --volume "$VOLUME" --layer penult --out "$OUT" \
  --isun --cifar10c

echo "=== 4. audit cache ==="
python -m extract.cache_audit --dir "$OUT" --audit --verify

echo "=== 5. exp0 (Path-3 gate) ==="
python -m experiments.exp0 --cache "$OUT"

echo "=== done. cache at $OUT ==="
