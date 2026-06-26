# First Real-Data Validation — RunPod

**What:** the clean-rebuild repo's Stage A ran end-to-end on **real** data for the first
time (RunPod RTX 4090, `chenyaofo cifar10_resnet20`, penult feat_dim 64). Five findings
emerged — **all invisible on synthetic**, because synthetic made disturbances
energy-*increasing* while real corruption/near-OOD does the opposite.

**Datasets:** CIFAR-10 (clean frame), CIFAR-100 (near-OOD / type-a), CIFAR-10-C
(corruption: fog, gaussian_noise, motion_blur). All real, on the network volume.

---

## Pipeline that ran

```bash
# populate (mount-agnostic manifest) — CIFAR-10/100/SVHN auto, CIFAR-10-C manual
python -m extract.populate_data --volume /runpod-volume

# layer sweep — which layer is best per axis
python -m extract.layer_sweep --data-root /runpod-volume/datasets/cifar10 --download

# extract 7 axes -> point cache + stream cache (provenance-tracked)
python -m extract.extract --volume /runpod-volume --layer penult \
    --out /runpod-volume/cache/run2 --cifar10c

# audit + Path-3 gate
python -m extract.cache_audit --dir /runpod-volume/cache/run2 --audit --verify
python -m experiments.exp0 --cache /runpod-volume/cache/run2
```

Infra worked: mount-agnostic manifest resolved (volume matched), cache provenance/audit
clean (3 real caches, 0 orphan). CIFAR-10-C needed the **full Zenodo tar** (record
2535967, 2.9 GB) — individual `.npy` URLs return 14 KB HTML 404s; verify by
`Content-Length > 1 GB` before downloading.

---

## Finding 1 — Path-3 gate PASSES on real data

**Experiment:** `exp0` sweeps the off-diagonal |correlation| threshold of the axis
covariance; curve shape (plateau vs collapse) tells whether the axis space is curved
(topology meaningful) or diagonal (axes independent, no topology). Re-run on **clean**
axis vectors with a larger sample to test if the plateau is real or noise.

**Result:**
| sample | bootstrap spread | verdict |
|--------|------------------|---------|
| 80 vectors (batch 32) | 0.40 | plateau |
| 320 vectors (batch 8) | **0.27** | plateau |

max off-diagonal corr = 0.63. The spread **dropped** as sample grew while the plateau
held → the curvature is **real structure, not small-sample noise**. The alignment-topology
layer is meaningful on real data (necessary condition met; "topology is *useful*" is a
separate, still-open question). *(4 single-batch axes only; sequence axes added below.)*

---

## Finding 2 — Axis correlation: two groups (DEVIATION ⊥ the rest)

**Experiment:** correlation matrix of the 4 single-batch axes on real clean activations
(320 vectors).

```
                 DEVIATION  CONSENSUS  CLUST_DIST  SUBNET_CONS
DEVIATION          1.00      -0.32       0.27        -0.55
CONSENSUS         -0.32       1.00       0.54         0.68
CLUSTER_DISTANCE   0.27       0.54       1.00         0.62
SUBNET_CONSENSUS  -0.55       0.68       0.62         1.00
```

**Result:** **DEVIATION is orthogonal** (negative corr) to the other three; the structure
axes (CONSENSUS / CLUSTER_DISTANCE / SUBNET_CONSENSUS) are mutually **positive** (0.54–0.68).
Two-group structure: **energy axis vs structure/alignment axes**. The positive correlation
among the three is *clean-state co-quiet* (all low when aligned), **not redundancy** — they
split on disturbances (shown synthetically: confidence_split separates CONSENSUS 0.50 from
SUBNET_CONSENSUS 0.97).

---

## Finding 3 — DRIFT_COH vs CLUST_DRIFT: false redundancy RESOLVED → complementary

**Experiment:** the two drift axes co-fired on synthetic (suspected redundant; kept both to
Stage A per the false-redundancy rule). Now measured on **real** CIFAR-10-C streams, block
order (sustained corruption).

**Result:**
| stream (block) | DRIFT_COH | CLUST_DRIFT |
|----------------|-----------|-------------|
| fog | 0.019 | **-0.59** |
| gaussian_noise | -0.033 | **-0.99** |
| motion_blur | 0.019 | **-0.98** |

**DRIFT_COH stays ~0 across all corruptions; CLUST_DRIFT varies by corruption type.** They
measure **different things → complementary, not redundant.** Pruning CLUST_DRIFT on synthetic
co-firing would have lost the corruption-type-discriminating signal. *This is the payoff of
the false-redundancy rule.*

(Sign note: CLUST_DRIFT is **negative** on real corruption — the distance-vector zig-zags;
block more negative than shuffle. Echoes the known "drift-coherence sign flips between
domains" lesson; sign interpretation still open.)

---

## Finding 4 — The energy-DECREASE discovery (the root cause)

**Experiment:** measure penult activation energy on clean vs corruption directly, then test
three DEVIATION variants (up = current one-sided / 2-side / down = decrease-only) as
AUC(clean vs disturbance).

**Energy at penult:**
```
clean          0.748
fog            0.726  (0.97x)   LOWER
gaussian_noise 0.504  (0.67x)   LOWER
motion_blur    0.592  (0.79x)   LOWER
```

**DEVIATION variants (AUC):**
| disturbance | up (current) | 2-side | down |
|-------------|--------------|--------|------|
| cifar100 (type-a) | 0.31 | 0.63 | **0.77** |
| fog | 0.41 | 0.51 | 0.60 |
| gaussian_noise | 0.28 | 0.75 | **0.86** |
| motion_blur | 0.34 | 0.59 | 0.72 |

**Result:** real corruption/near-OOD **lower** energy, not raise it. This is *why* DEVIATION
reads 0.31 (reversed) at penult and *why* PERSIST was all-zero. The `up` (current one-sided)
variant catches **nothing** (all reversed); `down` catches **everything**.

**Critical:** `down` catches type-a (0.77) **and** corruption equally → energy-decrease does
**not** separate type-a from corruption. So reverting DEVIATION to two-sided would
**reintroduce the type-a contamination** that motivated going one-sided originally.

**Decision: keep DEVIATION ONE-SIDED.** It not catching corruption is **correct
role-division, not a bug** — the structure axes already catch it (layer sweep penult:
CONSENSUS 0.73, CLUSTER_DISTANCE 0.76, SUBNET_CONSENSUS 0.86). DEVIATION stays the narrowest
axis: rare dangerous energy-*explosion* glitches only.

---

## Finding 5 — PERSIST must be decoupled from energy (it's a meta-axis)

**Experiment:** build a block stream (4 clean batches + 4 sustained gaussian_noise), compute
PERSIST on energy (current) vs on distance.

**Result:**
```
              clean(4)                sustained corrupt(4)
energy:   -0.02 0.06 0.01 -0.04  |  -0.39 -0.39 -0.39 -0.39   (never crosses +0.5)
distance:  1.04 1.10 1.07 1.09   |   1.41 1.40 1.38 1.39      (clearly rises)

PERSIST(energy)   = 0.00   <- the all-zero problem
PERSIST(distance) = 0.50   <- correctly catches the sustained block
```

Corruption **lowers energy but raises distance**, so energy-based PERSIST is blind while
distance-based PERSIST catches the sustained corruption. **PERSIST's persistence measure must
ride on the aggregate anomaly signal (distance/structure), not energy.** This confirms
PERSIST (and likely DRIFT_COH) are **meta-axes**: they read the *temporal pattern* of anomaly
independent of *which* axis fires. DRIFT_COH likely needs the same decoupling.

---

## Layer finding

Layer sweep verdict = **multi-layer worth it**: DEVIATION peaks at **layer2**, while
CONSENSUS / CLUSTER_DISTANCE / SUBNET_CONSENSUS peak at **layer3/penult**. (Consistent with
Finding 2: the energy axis lives at a different depth than the structure axes.)

---

## The throughline

Every finding traces to one fact synthetic hid: **real corruption/near-OOD lowers activation
energy; synthetic raised it.** Because we built synthetic disturbances energy-increasing,
DEVIATION (one-sided up) and PERSIST (energy-based) *looked* fine — but on real data they're
blind to the dominant disturbance direction. Real data forced two definition changes
(DEVIATION stays one-sided by *choice* now, not accident; PERSIST/DRIFT_COH decouple from
energy) and resolved one false-redundancy (DRIFT_COH ⊥ CLUST_DRIFT). The synthetic stand-in
validated the *logic*; real data corrected the *physics*.

---

## Next session

1. **Axis redesign** from these findings: one-sided DEVIATION (confirmed); PERSIST/DRIFT_COH
   re-based on the aggregate anomaly signal; resolve sequence-axis sign conventions.
2. **iSUN** for the DISTANCE-vs-DEVIATION energy-normal-type-b separation — the last
   unresolved false-redundancy pair.
3. **7-axis exp0** once sequence axes are re-based — does the plateau hold with all 7?
4. Fix `run_pod.sh` (`--isun --cifar10c` should be opt-in, not default).
