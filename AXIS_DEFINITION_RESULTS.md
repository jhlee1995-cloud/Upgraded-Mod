# 6-Axis Extraction — Definition Methodology & Results

**Context:** new clean repo (`modulator/`), rebuilding 6-axis extraction from scratch
(the old repo's extraction/gate/harness were entangled, so bugs were untraceable). The
6-axis *definitions* are kept (verified asset); only the *implementation* is rebuilt, with
each axis in an independent file + unit test so bugs stay isolated.

**Method:** for each axis, write MULTIPLE candidate formulas and choose by **selectivity** —
the axis should fire on ITS disturbance and stay quiet on the others. Selectivity is measured
on synthetic activations `(batch, 64)` (ResNet20 penultimate shape) via a **cross-reaction
matrix**: AUC(clean vs each disturbance) per candidate.

**Key methodological choice — mixtures, not just pure disturbances.** Testing pure
disturbances alone makes every candidate look fine (all hit AUC 1.0 on their target). Adding
*mixtures* (energy+cluster_b, structure+cluster_a, …) exposes **cross-reaction**: whether a
candidate leaks onto another axis's target. This repeatedly drove the decisions below.

**type-b handling (the Stage-B lesson, carried forward):** type-b cannot be cleanly
synthesized. It is flagged synthetic-unreliable and its final validation deferred to Stage A
real near-OOD (CIFAR-100/iSUN). Synthetic type-b numbers here are qualitative only.

---

## Disturbance model (activation level)

| disturbance | construction | axis it should trigger |
|-------------|--------------|------------------------|
| `energy` | norm inflation (×2.5) | DEVIATION |
| `structure` | independent per-channel-group push **larger than inter-center distance** (so votes actually flip) | CONSENSUS |
| `cluster_a` (type-a) | move to midpoint between two centers (between clusters) | CONSENSUS / SUBNET-CONSENSUS |
| `cluster_b` (type-b) | move near a WRONG center, **norm preserved** (energy normal) | CLUSTER-DISTANCE only |
| mixtures | two disturbances applied in sequence | tests selectivity |

Subnets = **channel-group split** (width, K=4 groups of 16), NOT depth heads (depth heads are
dead, AUC~0.5 — known).

---

## Results per axis

### 1. DEVIATION — `l2_signed`  (selectivity 0.42)

Normalized activation-energy deviation. Five candidates tested.

| candidate | target (energy) | cross-leak | selectivity |
|-----------|-----------------|-----------|-------------|
| l2_zscore | 1.00 | 0.80 | 0.20 |
| l2_robust | 1.00 | 0.80 | 0.20 |
| **l2_signed** | **1.00** | **0.58** | **0.42** |
| per_channel | 1.00 | 0.77 | 0.23 |
| per_channel_max | 1.00 | 0.78 | 0.22 |

**Finding:** all catch energy perfectly, but **two-sided** formulas (|norm − μ|) wrongly fire
on `cluster_a` (AUC 1.0) because type-a *reduces* norm (midpoint) and two-sided deviation
counts deflation as anomaly. The **signed** formula (inflation only, deflation→0) ignores
type-a and type-b — selectively catching only energy *inflation*. This matches "DEVIATION is
the tightest signal": signed = narrow, catches only what it should. *Mixtures with cluster_a
were what exposed this; pure energy alone showed all candidates at 1.0.*

### 2. CONSENSUS — `disagree_rate`  (selectivity 0.94)

Agreement among K channel-group subnets (hard vote = nearest center).

| target (structure & cluster_a) | non-target leak (energy, cluster_b) | selectivity |
|--------------------------------|-------------------------------------|-------------|
| 0.94 | 0.00 | 0.94 |

(all four candidates — disagree_rate / majority_gap / logit_variance / vote_entropy — tied;
chose the simplest, `disagree_rate`.)

**Finding:** fires on `structure` (0.92) and `cluster_a` (type-a, 0.95); **exactly 0.50 on
energy and cluster_b**. The cluster_b 0.50 is *correct, not a bug* — type-b's votes don't
split (all subnets agree on the wrong center), the known fundamental blind spot.

**Diagnostic that saved a correct formula:** initially `structure` scored only 0.51. A
strength sweep showed the synthetic `structure` push (4.0) was *smaller* than inter-center
distance (6.0), so votes never flipped — a **weak-synthetic** problem, not a formula defect.
Raising the push above center-separation restored 0.92. (Sweep: strength 4→0.51, 8→0.73,
12→0.92, 24→0.99.) Correct CONSENSUS behavior is to ignore sub-threshold breaks.

### 5. SUBNET-CLUSTER-DISTANCE — `mean_nearest`  (the type-b detector)

Mean per-subnet nearest-center distance. Four candidates + an oracle diagnostic.

**type-b diagnosis (the crux):**
```
oracle (true-center distance) AUC = 1.00   <- type-b IS displaced (ceiling if label known)
best label-free candidate         = 0.75   <- label-free can only partially catch it
```

**Finding:** the 1.00-vs-0.75 gap *is* the project's central hard problem. type-b is genuinely
displaced (oracle proves it), but label-free detection is partial because type-b sits *near a
(wrong) center* — small nearest-distance, looks clean. Among label-free candidates, distance is
the ONLY one with any type-b signal (0.75 vs ~0.50 for all vote-based axes), confirming "type-b
is caught only by distance." Distance is **non-selective** (fires on energy/structure/cluster_a
all at 1.0 too) — its role is not selectivity but being the sole partial type-b channel.
**The synthetic 0.75 is not trustworthy; the real gap size is a Stage-A question (CIFAR-100).**

Consequence for the design: type-b is specified by a *combination* — distance-high AND
consensus-low (votes don't split) — not by any single axis.

### 6. SUBNET-CONSENSUS — `mean_entropy`  (complementary to CONSENSUS, verified)

Soft-distribution agreement among subnets, contrasted with the 2nd axis (hard vote).
To test complementarity we added a disturbance — `confidence_split` — where subnets vote for
the SAME center (hard vote identical) but disagree on CONFIDENCE (soft spread).

**Complementarity verdict (the `confidence_split` column):**
```
2nd CONSENSUS (hard)        on confidence_split: 0.50   <- blind (votes identical)
6th js_divergence  (soft)   on confidence_split: 0.91   -> COMPLEMENTARY
6th confidence_var (soft)   on confidence_split: 0.86   -> COMPLEMENTARY
6th mean_entropy   (soft)   on confidence_split: 0.97   -> COMPLEMENTARY
```

**Finding:** the hard-vote axis is **completely blind** (0.50) to a case the soft axis catches
(0.97). This directly verifies "neither subsumes the other" — not assumed, *measured*. The 6th
is more sensitive but less selective (fires on most disturbances when confidence wobbles); the
2nd is selective (only true vote-splits). Their division of labor: 2nd = precise detector of
*severe* anomalies (vote split), 6th = sensitive detector of *subtle* instability (confidence
wobble). Both remain type-b blind (0.55–0.60), consistent with the known structure.

---

## Cross-reaction summary (the orthogonality picture)

|  | energy | structure | cluster_a | cluster_b (type-b) |
|--|--------|-----------|-----------|---------------------|
| **DEVIATION** (l2_signed) | **1.00** | 0.95* | 0.27 | 0.50 |
| **CONSENSUS** (disagree) | 0.50 | **0.92** | **0.95** | 0.50 |
| **CLUSTER-DIST** (mean_near) | 1.00† | 1.00† | 1.00† | **0.75** |
| **SUBNET-CONS** (entropy) | 0.93 | **1.00** | **1.00** | 0.60 |

\* DEVIATION leaks slightly onto structure (structure_break also raises norm a bit) — to be
re-checked at Stage A. † CLUSTER-DISTANCE is intentionally non-selective (catches any
off-center displacement); its unique value is the type-b column.

The diagonal is alive (each axis responds to its target) and **type-b is the common blind
spot across all vote/energy axes** — only distance has partial signal. This reproduces the
known signal structure on synthetic data.

---

## Decisions (provisional, pending comparison with the old implementation)

| axis | chosen formula | target | key finding |
|------|----------------|--------|-------------|
| 1 DEVIATION | `l2_signed` | energy inflation | one-sided avoids type-a contamination |
| 2 CONSENSUS | `disagree_rate` | structure, type-a | hard vote; type-b blind (correct) |
| 5 CLUSTER-DISTANCE | `mean_nearest` | type-b | sole partial detector (0.75); Stage-A validation required |
| 6 SUBNET-CONSENSUS | `mean_entropy` | confidence instability | complementary to 2nd (0.97 vs 0.50 on confidence_split) |

**Deferred to Stage A (sequence functions, not testable on static synthetic):**
- **3 PERSISTENCE** — consecutive-anomalous-batch streak
- **4 DRIFT-COHERENCE** — cosine coherence of recent push-directions

(Same limitation found in Module 4: time/trajectory axes need real-time streams.)

---

## Next

Compare these four chosen formulas against the **old repo's implementation** — did the old
code already handle the cross-contamination (e.g. one-sided DEVIATION, channel-group subnets),
or does the rebuild improve selectivity? Then wire the chosen axes into the Stage-A extractor
(activation → 6-axis → cache), keeping the Stage-B cache contract intact.
