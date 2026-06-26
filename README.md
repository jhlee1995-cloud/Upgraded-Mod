# Modulator

Label-free, gradient-free control-plane modulator for neural-network test-time
adaptation. Reads only activation statistics (no loss, gradients, or labels) to detect
**model-internal-topology ↔ data misalignment**, and gates both adaptation and
action/fallback decisions.

This is a **clean rebuild** of an earlier prototype, split into auditable layers so bugs
stay isolated. The 6→7 axis *definitions* are kept (verified asset); only the
implementation is rebuilt, each axis in its own file with cross-reaction validation.

---

## Architecture: two stages

The key structural fact: **axis vectors depend only on (data, layer) — not on order or
seed.** So GPU forward passes are done once and cached; everything else is cheap CPU work
on the cache.

```
STAGE A  (extract/)   GPU, on RunPod        STAGE B  (frame/)   CPU, anywhere
  data → backbone → 7 axes → cache    ──►    cache → coordinate frame → topology
  (the expensive part, run once)             (the brain, runs on the cache)
```

The two stages are connected ONLY by the cache contract (provenance-tracked `.npy`).
Stage B never knows whether a cache is synthetic or real — so logic validated on a
synthetic stand-in runs unchanged on real data.

---

## The 7-axis pool

Formulas chosen by **cross-reaction selectivity** (each axis fires on its disturbance,
quiet on others), validated on synthetic activations/streams. See
`AXIS_DEFINITION_RESULTS.md` and `SEQUENCE_AXIS_RESULTS.md`.

| # | axis | formula | target | kind |
|---|------|---------|--------|------|
| 1 | DEVIATION | `l2_signed` | energy inflation | single-batch |
| 2 | CONSENSUS | `disagree_rate` | structure / type-a | single-batch |
| 3 | CLUSTER_DISTANCE | `mean_nearest` | type-b (sole detector) | single-batch |
| 4 | SUBNET_CONSENSUS | `mean_entropy` | confidence instability | single-batch |
| 5 | DRIFT_COH | `window_cosine` | directional drift | sequence |
| 6 | PERSIST | `streak` | sustained vs transient | sequence |
| 7 | CLUST_DRIFT | `consec_cosine` | anchored drift | sequence |

**Redundancy questions are OPEN — kept to Stage A.** The "false redundancy" trap (synthetic
side-effects make two signals co-fire) means redundancy can only be confirmed on real data
with the *separating* input. Both members of each suspected pair stay in the pool:
- DRIFT_COH vs CLUST_DRIFT — separate only when distance-vector moves but batch-mean fixed
- CLUSTER_DISTANCE vs DEVIATION — separate only on energy-normal near-OOD (iSUN)

Never prune on synthetic co-firing.

---

## Pod workflow

```bash
# 0. install
pip install -r requirements.txt          # or: docker build -t modulator .

# 1. populate datasets ONCE (mount-path agnostic via manifest)
python -m extract.populate_data --volume /runpod-volume
#    CIFAR-10/100/SVHN auto-download; CIFAR-10-C and iSUN are manual (manifest tells you)

# 2. decide the extraction layer (single vs multi)
python -m extract.layer_sweep --data-root /runpod-volume/datasets/cifar10 --download

# 3. extract 7 axes → provenance-tracked cache (point + stream)
python -m extract.extract --volume /runpod-volume --layer penult \
    --out /runpod-volume/cache/run1 --isun --cifar10c

# 4. audit the cache (provenance, real/synthetic, contract)
python -m extract.cache_audit --dir /runpod-volume/cache/run1 --audit --verify

# 5. Stage B: the Path-3 gate
python -m experiments.exp0 --cache /runpod-volume/cache/run1
```

`run_pod.sh` chains these.

---

## Layout

```
extract/                    Stage A (GPU): activation → 7 axes → cache
  axes/                       7 axis formulas, one file each (+ cross-reaction tests)
  synth_activations.py        synthetic stand-in (single-batch) for logic validation
  synth_streams.py            synthetic stand-in (sequence) for logic validation
  backbone.py                 backbone + multi-layer hooks
  axis_registry.py            the 7 chosen formulas behind one interface
  data_loaders.py             CIFAR-10-C, iSUN, stream builder
  layer_sweep.py              per-axis best-layer measurement
  extract.py                  ties it together → point + stream caches
  populate_data.py            one-time dataset population + manifest
  cache_audit.py              provenance-tracked cache layer
frame/                      Stage B (CPU): cache → coordinate frame → topology
  cache.py                    Stage B cache reader
  coordinate.py               multi-center frame (k-sweep, Mahalanobis, branch metrics)
  selfconsistency.py          tuning vs missed-axis discriminator
experiments/                exp0 (Path-3 gate), exp1, scale_gate, axis_selection
analysis/                   plots, reports
tests/                      per-axis + frame unit tests
cache/                      cache output (gitignored)
```

---

## Validation status

| component | validated on | status |
|-----------|--------------|--------|
| 7 axis formulas (single-batch) | synthetic activations + cross-reaction | ✓ |
| 7 axis formulas (sequence) | synthetic streams + cross-reaction | ✓ |
| coordinate frame (k-sweep, severity, mass) | synthetic ground truth | ✓ |
| self-consistency (tuning/missed-axis) | synthetic ground truth | ✓ |
| Stage A pipeline (backbone→axes→cache) | end-to-end on torch | ✓ |
| populate (mount-agnostic) + cache audit | full chain | ✓ |

**Deferred to Stage A real data (synthetic-unverifiable, by design):**
1. type-b detection — synthetic type-b leaks energy; needs CIFAR-100/iSUN
2. trajectory mode — drift needs real-time movement; static synthetic has none
3. DRIFT_COH/CLUST_DRIFT and CLUSTER_DISTANCE/DEVIATION redundancy — need separating inputs
4. axis-set finalization — 7-candidate pool pruned by real cross-reaction + gradient-Gram eff-dim

---

## Design principles (enforced)

- 6→7 axes are the fixed measurement basis — measure on top, never re-derive or drop.
- Every threshold is **swept as a curve**, never pinned (the shape is threshold-independent).
- Clean is the **ruler** (coordinate frame), removed from the graph; only misalignment is drawn.
- The frame is **time-invariant**; updates append clusters; global re-bake is discrete and rare.
- **Verify before recording** — worries AND redundancy. Never prune on synthetic co-firing.
- Synthetic only validates logic; real data decides. Synthetic ceilings are the recurring trap.
```
