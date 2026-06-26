# Sequence-Axis Definition — Results

**Context:** continuing the new-repo rebuild. After comparison with the old notebook
(`six_axis_real_data`), we adopted approach **(c)**: don't assume 6 axes — extract a
**7-candidate pool** (4 single-batch + 3 sequence) and let real data prune via cross-reaction
AND gradient-Gram eff-dim. This document covers defining the **3 sequence axes**, which cannot
be validated on static synthetic data (the Module-4 wall).

**Method:** plant TIME structure into synthetic streams (the time-dimension analogue of the
single-batch disturbance approach), logic-validate candidate formulas here, defer final
selection to Stage A real streams.

---

## Planted-time-structure streams

A stream = T feature batches `[h_1 … h_T]`, each `(B, 64)`. Energy trajectories confirm the
planted structure:

| stream | energy trajectory (T=8) | what's planted |
|--------|-------------------------|----------------|
| `clean` | 36 36 37 36 38 38 37 37 | fixed distribution (no signal) |
| `drift` | 37 36 38 39 40 41 43 43 | mean shifts one direction (energy flat — drift is *direction*, not magnitude) |
| `persistent` | 226 224 232 226 235 236 231 229 | anomaly in EVERY step |
| `transient` | 36 36 37 36 **235** 38 37 37 | anomaly in ONE step only |
| `clust_drift` | 37 30 27 24 22 19 17 18 | creep toward a wrong center (distance-vector shifts) |

---

## Candidate results (after diagnostics)

### DRIFT_COH — chosen `window_cosine` (improves on old notebook)

| candidate | drift (target) | persistent | transient | clust_drift | selectivity |
|-----------|----------------|------------|-----------|-------------|-------------|
| consec_cosine (old) | 0.86 | 0.49 | 0.64 | 1.00 | 0.43 |
| path_straight | 1.00 | 0.51 | 0.06 | 1.00 | 0.36 |
| **window_cosine** | **0.98** | 0.50 | 0.53 | 1.00 | **0.63** |

**Diagnostic that mattered:** initially DRIFT_COH scored 0.45 on its own target (below clean!).
A drift-magnitude sweep showed the synthetic drift (mag 3) was *smaller than batch noise*
(spread 1.0), so the directional trend was buried — weak-synthetic, not formula-defect (same
pattern as CONSENSUS earlier). After raising drift above noise, the candidates separated
sharply:

```
drift_mag:        3     6    10    20    40    80
consec_cosine:  0.00  0.00  0.00  0.00  0.82  1.00   <- noise-fragile (old formula)
window_cosine:  0.65  0.75  0.88  0.97  1.00  1.00   <- noise-robust
```

**Finding:** the old notebook's `consec_cosine` (mean cosine of consecutive mean-pushes) is
**noise-fragile** — batch-mean jitter makes consecutive pushes zig-zag, collapsing the cosine
to 0 even when drift is present. `window_cosine` (windowed-average pushes) averages out noise
and recovers the trend. In real (noisy) data, the old formula would likely *miss* drift; the
rebuild's window form is an improvement. (Only visible because we compared candidates.)

### PERSIST — chosen `streak` (improves on old notebook)

PERSIST's real job is **persistent-vs-transient**, not vs-clean (a transient spike differs from
clean too). Using that as the metric:

| candidate | persist-vs-transient AUC |
|-----------|--------------------------|
| autocorr (old) | **0.43** (fails — can't separate) |
| **streak** | **1.00** |
| fraction | 1.00 |

**Finding:** the old notebook's `autocorr` (lag-1 autocorrelation of energy deviation) **cannot
distinguish persistent from transient** (0.43). A longest-run `streak` (or above-threshold
`fraction`) separates them perfectly. Chose `streak`. Clear improvement over the old formula.

### CLUST_DRIFT — chosen `consec_cosine` (same as old)

| candidate | clust_drift (target) | drift | persistent | transient | selectivity |
|-----------|----------------------|-------|------------|-----------|-------------|
| **consec_cosine** | **1.00** | 1.00 | 0.54 | 0.59 | 0.58 |
| total_shift | 1.00 | 1.00 | 1.00 | 0.50 | 0.33 |
| straightness | 1.00 | 1.00 | 0.45 | 0.00 | 0.30 |

Old notebook's distance-vector consecutive-cosine works; kept.

---

## KEY FINDING — DRIFT_COH vs CLUST_DRIFT may be redundant (provisional)

The old notebook included BOTH as separate axes. Cross-reaction suggests they are **redundant
on this synthetic data**:

```
stream         |  DRIFT_COH  CLUST_DRIFT
drift          |    0.98       1.00       <- both fire
clust_drift    |    1.00       1.00       <- both fire
```

Complementarity would require DRIFT_COH to fire on `drift` but NOT `clust_drift` (and vice
versa). Instead **both fire on both** → redundant signal here. If this holds on real data, the
axis pool drops from 7 toward 6 (or 5) — drop one of the two.

**Honest caveat (defer to Stage A):** the synthetic `clust_drift` moves the mean toward a wrong
center, which *also* moves the batch-mean — so DRIFT_COH catches it as a side effect. A genuine
cluster-drift that moves the distance-vector while leaving the batch-mean fixed cannot be
synthesized here (akin to the type-b synthesis limit). So redundancy is **not confirmed** — it
requires a real stream where the two structures are genuinely separable. This is the **third**
"synthetic-unverifiable, Stage-A-required" item.

---

## The 7-candidate pool (formulas now fixed; selection pending real data)

| # | axis | chosen formula | vs old notebook | validation status |
|---|------|----------------|-----------------|-------------------|
| 1 | DEVIATION | `l2_signed` | old was two-sided (leaks onto type-a) → one-sided **improves** | synthetic ✓ |
| 2 | CONSENSUS-hard | `disagree_rate` | old merged hard+soft → **split** (hard) | synthetic ✓ |
| 3 | CLUSTER-DISTANCE | `mean_nearest` | same | synthetic ✓ (type-b → Stage A) |
| 4 | SUBNET-CONSENSUS-soft | `mean_entropy` | old merged hard+soft → **split** (soft); complementary verified | synthetic ✓ |
| 5 | DRIFT_COH | `window_cosine` | old `consec` noise-fragile → window **improves** | synthetic ✓ |
| 6 | PERSIST | `streak` | old `autocorr` can't separate persist/transient → **improves** | synthetic ✓ |
| 7 | CLUST_DRIFT | `consec_cosine` | same | synthetic ✓ (redundancy w/ #5 → Stage A) |

---

## Three items deferred to Stage A (synthetic-unverifiable, by design)

1. **type-b detection** — synthetic type-b leaks energy; real CIFAR-100/iSUN required.
2. **trajectory mode** — drift needs real-time movement; static synthetic has none.
3. **DRIFT_COH / CLUST_DRIFT redundancy** — separating the two needs a real stream where
   distance-vector moves but batch-mean doesn't.

These are not gaps in the rebuild — they are the precise questions only real data can answer,
now clearly isolated.

---

## Net comparison with the old notebook

The rebuild changed **4 of 7 formulas** for the better, all surfaced by the candidate-vs-
candidate cross-reaction method (the old notebook used single fixed formulas, so these
weaknesses were invisible):
- DEVIATION: two-sided → one-sided (avoids type-a contamination)
- CONSENSUS: one merged soft axis → **split into hard (#2) + soft (#4)**, verified complementary
- DRIFT_COH: noise-fragile consecutive → noise-robust windowed
- PERSIST: autocorr (can't separate) → streak (perfect separation)

The old notebook's **gradient-Gram eff-dim** measurement is *also* kept for Stage A — it answers
"how many independent dimensions" directly, complementing cross-reaction's "what does each axis
detect". Both run on the 7-candidate pool at Stage A to finalize the axis set.
