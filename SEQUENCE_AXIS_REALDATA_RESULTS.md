# Sequence-Axis Investigation — Real Data (RunPod)

**What:** after re-basing PERSIST/DRIFT_COH on aggregate anomaly (not energy) and storing
signed+abs drift forms, we tested the sequence axes on real CIFAR-10-C streams. A
**stream-builder bug** was found that had contaminated the previous session's results;
fixing it and adding a proper **ramp-severity** stream clarified what each sequence axis
actually detects.

**Setup:** `chenyaofo cifar10_resnet20`, penult, `/workspace` (persistent volume),
CIFAR-10-C (fog, gaussian_noise, motion_blur). PERSIST mode = distance. 40 streams per
condition.

---

## Decision → Evidence (what each decision was based on)

| decision | based on (experiment + numbers) |
|----------|----------------------------------|
| **PERSIST re-based on distance, not energy** | Energy at penult: corruption *lowers* it (gaussian 0.67×, motion 0.79×). Block stream (4 clean + 4 sustained gaussian): energy trajectory drops -0.39 (never crosses +0.5 → PERSIST(energy) = 0.00), but distance rises 1.09→1.41 → PERSIST(distance) = 0.50. Energy-based was blind; distance-based caught it. |
| **DEVIATION kept one-sided** | DEVIATION-variant AUC (clean vs disturbance): `down` (decrease) catches type-a (cifar100 0.77) AND corruption (0.60–0.86) **equally** → two-sided would re-introduce type-a contamination. Structure axes already catch corruption (CONSENSUS 0.73, CLUSTER_DISTANCE 0.76, SUBNET_CONSENSUS 0.86). So DEVIATION not catching corruption = correct role division. |
| **Stream-builder block mode rewritten** | Block produced `.C.C.C.C` (alternating, block_size = n//6 = 1), not sustained. This caused PERSIST=0 on block and the artificial CLUST_DRIFT −0.99. Fixed to `....CCCC`. |
| **Previous "CLUST_DRIFT discriminates corruption" RETRACTED** | With true sustained blocks, CLUST_DRIFT = −0.35 to −0.44 and block ≈ shuffle (gaussian −0.351 vs −0.368). The −0.99 was the `.C.C.C.C` zig-zag artifact, not a real signal. |
| **signed + abs drift both stored** | Synthetic predicted ramp → positive CLUST_DRIFT (+0.23); real ramp gave **−0.35** (prediction wrong). Storing both forms made the sign visible without re-extracting. Sign convention is unresolved ("drift coherence flips sign between domains"), so it's decided from data, not guessed. |
| **Ramp-severity stream added** | Sustained on/off corruption is not gradual drift. DRIFT_COH was weak on block (0.08–0.11). To test DRIFT_COH/CLUST_DRIFT on their actual target (directional gradual drift), severity ramps 1→5. |
| **DRIFT_COH validated as directional-drift detector** | ramp(linear) vs block, DRIFT_COH_signed: gaussian 0.391±0.087 vs 0.114±0.096 (std = ¼ mean → solid separation). |
| **DRIFT_COH "blind to motion_blur" judged correct, not a bug** | motion_blur ramp 0.093 vs block 0.080 (no separation), but gaussian ramp 0.391. motion_blur is non-directional (smear, no consistent direction); DRIFT_COH measures directional coherence by definition → legitimately not its target. |
| **CLUST_DRIFT put on probation** | Across all stimuli: ramp vs block (gaussian −0.353 vs −0.331) and block vs shuffle both show no separation. No selective signal of its own → independent usefulness questionable (candidate to drop 7→6, pending a unique-coverage test). |

---

# Detailed findings

---

## The stream-builder bug (and retraction)

**Bug:** `build_stream_batches` block mode used `block_size = n_steps // 6 = 1` for
`stream_len=8`, producing **alternating singletons** `.C.C.C.C` instead of a **sustained**
corruption run. This invalidated the previous session's drift conclusions.

**Fix:** a single contiguous corrupt run at the stream end, starting clean (`....CCCC`),
sized by ratio — so PERSIST's baseline (first third of stream) is valid and the corruption
is genuinely sustained.

**Retraction:** the previous claim "CLUST_DRIFT varies by corruption type (gaussian -0.99 vs
fog -0.64), catches corruption" was a **bug artifact** — the `.C.C.C.C` extreme zig-zag of
the distance-vector gave an artificially strong negative cosine. With true sustained blocks,
CLUST_DRIFT is -0.35 to -0.44 and **block ≈ shuffle** (doesn't discriminate). The
"DRIFT_COH ⊥ CLUST_DRIFT confirmed" conclusion rested on that artifact and was retracted.

*Lesson: the measurement instrument (the stream builder) is itself subject to "verify before
recording." A bug there flipped a downstream conclusion.*

---

## PERSIST — validated, statistically solid

Re-based on distance (corruption lowers energy but raises distance, so energy-based PERSIST
was blind — all zeros). On sustained blocks:

| corruption | block | shuffle |
|------------|-------|---------|
| gaussian_noise | **0.517 ± 0.022** | 0.038 ± 0.078 |
| motion_blur | **0.510 ± 0.038** | 0.031 ± 0.100 |
| fog | 0.152 ± 0.133 | 0.065 ± 0.116 |

gaussian/motion: block ≫ shuffle, std ≈ 1/20 of mean → **near-perfect separation.** fog is
weak *because fog barely lowers energy* (0.97×) so distance barely moves — **correct
strength-proportional behavior, not a defect.** PERSIST scales with corruption strength.

---

## Ramp-severity streams — the proper drift test

Sustained on/off corruption is the **wrong** stimulus for DRIFT_COH/CLUST_DRIFT — their
target is *gradual directional drift*. So we built `build_ramp_stream`: severity ramps 1→5
over the stream (linear: even progression; late: low then ramp), making genuine gradual
drift.

```
linear: severity = 1 1 1 2 2 3 3 3 4 4 5 5
late:   severity = 1 1 1 1 1 1 1 1 2 3 4 5
```

---

## The clarified 3-way role picture

### PERSIST — sustained detector, direction-AGNOSTIC
Catches persistence whether gradual or abrupt (ramp PERSIST also elevated 0.16–0.40).
Strong, solid. Its job: *is the anomaly continuing?*

### DRIFT_COH — directional gradual-drift detector (direction-specific)

**Corruption-dependent — the key finding.** ramp(linear) vs block, DRIFT_COH_signed:

| corruption | ramp(linear) | block | separation |
|------------|--------------|-------|------------|
| gaussian_noise | **0.391 ± 0.087** | 0.114 ± 0.096 | **sharp (std = ¼ mean)** |
| fog | 0.112 ± 0.060 | 0.047 ± 0.081 | weak |
| motion_blur | 0.093 ± 0.075 | 0.080 ± 0.098 | **none** |

**Interpretation:** gaussian_noise is a **directional** corruption — noise pushes activations
consistently one way, so the batch-mean drifts directionally and DRIFT_COH catches it (0.39).
motion_blur is **non-directional** — blur smears without a consistent direction, so the
batch-mean doesn't drift coherently and DRIFT_COH is legitimately blind (0.09). This is
**correct behavior**: DRIFT_COH measures directional coherence by definition, so
non-directional drift is not its target. (linear ramp > late ramp: 0.391 vs 0.210 for
gaussian — more consistent direction = stronger signal.) Its job: *is the anomaly drifting
in a consistent direction?*

### CLUST_DRIFT — role unclear, weak selectivity
Always negative (-0.3 to -0.44). Does **not** distinguish ramp vs block (gaussian -0.353 vs
-0.331), **nor** block vs shuffle. It shows no selective signal of its own across any
stimulus tested — sustained, gradual, or directional. Its independent usefulness is
**questionable**.

*(The signed/abs storage paid off: the synthetic-based prediction "ramp → positive
CLUST_DRIFT" (synthetic gave +0.23) was **wrong** on real data (real ramp gave -0.35). Real
distance-vector geometry is more complex than synthetic — severity rise moves the
distance-vector components erratically, not coherently. Another synthetic-intuition corrected
by real data; storing both forms let us see it.)*

---

## Where this leaves the axes

| axis | detects | status |
|------|---------|--------|
| PERSIST | persistence (direction-agnostic) | ✓ solid (block 0.51 vs shuffle 0.03) |
| DRIFT_COH | directional gradual drift | ✓ validated on directional corruption (gaussian 0.39); blind to non-directional (motion 0.09) — correct |
| CLUST_DRIFT | ? | ✗ no selective signal found; usefulness questionable |

**PERSIST ↔ DRIFT_COH complementarity is now clear and well-grounded** — one detects
persistence (direction-agnostic), the other directional drift (direction-specific). Genuinely
different temporal structures.

**DRIFT_COH vs CLUST_DRIFT** (the last false-redundancy pair) is **partially resolved**: not
redundant (different sign/geometry; DRIFT_COH selectively catches directional drift), but
CLUST_DRIFT shows no selective signal of its own → it may be the droppable axis (pool 7→6).
**Open question:** does CLUST_DRIFT uniquely catch anything PERSIST + DRIFT_COH miss? Needs a
dedicated test before dropping.

---

## Throughline

Two synthetic-vs-real corrections this session, both caught by good instrumentation:
1. **Stream-builder bug** — alternating ≠ sustained; fixing it retracted a drift conclusion.
2. **Sign prediction wrong** — synthetic said ramp→positive CLUST_DRIFT; real said negative.
   Storing signed+abs (instead of guessing the sign) made it visible.

The sequence axes are no longer a black box: PERSIST (persistence) and DRIFT_COH (directional
drift) have clear, validated, complementary roles; CLUST_DRIFT is on probation.

---

## Next

1. **iSUN** for distance ⊥ deviation — the last single-batch false-redundancy pair.
2. **7-axis exp0** with the re-based sequence axes — does the plateau hold with all 7?
3. **CLUST_DRIFT unique-coverage test** — does it catch anything the others miss? If not, drop.
4. **branch / arrangement engine** — only after the axis set is settled.
