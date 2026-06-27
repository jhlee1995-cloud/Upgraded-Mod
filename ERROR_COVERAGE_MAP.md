# Error-Coverage Map — Master Tracking Document

**Goal:** fill every cell, and keep evidence honestly separated from expectation. Rows are
*error mechanisms* (mechanism determines which axis fires). Columns are *detection axes*.
This document is a **decision engine** (see Escalation), not just a record.

---

## Legend (disambiguated)

**Detection status:**
- ✅ = experimentally verified on real data (signal present)
- 🟢 = theoretically expected, **not yet verified** (an expectation, not a fact)
- 🟡 = verified on **synthetic only** (real test pending)
- ✗ = tested on real data, **no signal** (and understood why)
- — = conceptually not applicable (this axis cannot, by construction, see this mechanism)
- ⬜ = not tested

**Why 🟢 vs ✅ matters:** the project's recurring failure is assumptions hardening into
"facts" (the false-redundancy trap). A cell may NOT be ✅ until a real experiment measured
it; until then it is at most 🟢. Never silently promote 🟢→✅.

**Confidence (independent of status — how much evidence backs the row's mapping):**
- HIGH = multiple real-data experiments agree
- MEDIUM = one real experiment
- LOW = synthetic only
- UNKNOWN = not tested

---

## The Map (strictly mechanism × axis; severity lives in its own matrix below)

Axes: DEV=DEVIATION · CON=CONSENSUS · CLD=CLUSTER_DISTANCE · SUB=SUBNET_CONSENSUS ·
DCH=DRIFT_COH · PER=PERSIST · CDR=CLUST_DRIFT

| # | error mechanism | DEV | CON | CLD | SUB | DCH | PER | CDR | primary metric | confidence |
|---|-----------------|-----|-----|-----|-----|-----|-----|-----|----------------|------------|
| 1 | energy explosion | 🟡 | — | — | — | — | — | — | clean-vs-stimulus AUC (energy↑) | LOW |
| 2 | energy decrease | ✗ | ✅ | ✅ | ✅ | — | — | — | clean-vs-corruption AUC | HIGH |
| 3 | between-cluster (type-a) | ✗ | ✅ | 🟢 | ✅ | — | — | — | clean-vs-type-a AUC; vote-split rate | MEDIUM |
| 4 | wrong-cluster (type-b) | ✗ | 🟢 | ✅ | 🟢 | — | — | — | per-sample correct-vs-confwrong AUC (margin best 0.895) | MEDIUM |
| 5 | directional gradual drift | — | — | — | — | ✅ | ✅ | ✗ | ramp vs block (DRIFT_COH sep>1.4) | HIGH |
| 6 | non-directional gradual drift | — | — | — | — | ✗ | 🟡 | ✗ | ramp vs block; PER covers fog/snow, GAP brightness/defocus/motion | MEDIUM |
| 7 | sustained disturbance | — | — | — | — | — | ✅ | ✗ | block vs shuffle (PERSIST streak) | HIGH |
| 8 | adversarial | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | clean-vs-FGSM/PGD AUC | UNKNOWN |
| 9 | recovery (drift returning) | — | — | — | — | ✅ | 🟢 | ✗ | recovery vs ramp signed-drift sep 1.99 (DRIFT_COH_signed) | MEDIUM |

*(Row "very weak disturbance" removed from the mechanism map — that's a severity level, not
a mechanism. It lives in the Detection-Threshold Matrix below. See taxonomy rules Freeze.)*

---

## Detection-Threshold Matrix (mechanism × severity — a SEPARATE dimension)

Difficulty is orthogonal to mechanism. This matrix tracks, per mechanism, the severity at
which detection holds (escaping the AUC ceiling that saturated everything at severity-3).

Per-axis detection floor (lowest severity caught, AUC>0.7), from the severity sweep:

| corruption | DEVIATION | CONSENSUS | CLUSTER_DIST | SUBNET_CONS |
|------------|-----------|-----------|--------------|-------------|
| brightness | — | — | — | s3 |
| contrast | s1 | s2 | s2 | s1 |
| defocus_blur | s2 | s2 | s2 | s2 |
| fog | s2 | s2 | s3 | s1 |
| gaussian_noise | s1 | s1 | s1 | s1 |
| motion_blur | s1 | s1 | s1 | s1 |
| pixelate | s1 | s1 | s1 | s1 |
| snow | s1 | s1 | s1 | s1 |

(— = not caught through s3.) **SUBNET_CONSENSUS is the most sensitive axis** (uniquely
catches brightness; fog/contrast at s1); **CLUSTER_DISTANCE is the least sensitive** (fog only
at s3). Different floors per axis = direct evidence the axes are NOT redundant.

At severity-3 almost all axes hit AUC 1.00 (ceiling) → the floor (lowest severity still
detected) is unknown. The severity-1 sweep fills this matrix and is where axis *differences*
become visible (saturation hides them).

---

## Per-Axis Unique Value (the canonical reason to KEEP each axis)

| axis | unique contribution | status of that claim |
|------|---------------------|----------------------|
| DEVIATION | only detector for energy *explosion* (one-sided, inflation) | 🟡 synthetic; no real explosion stimulus |
| CONSENSUS | vote-split / between-cluster (type-a) | ✅ real (but co-fires with CLD/SUB on type-a) |
| CLUSTER_DISTANCE | **only candidate** for type-b (wrong-cluster) | ⬜ load-bearing IFF it fires on real type-b |
| SUBNET_CONSENSUS | soft confidence instability (entropy) | ✅ real (entangled with CON/CLD at 0.6) |
| PERSIST | sustained anomaly over time | ✅ real, solid (block 0.51 vs shuffle 0.03) |
| DRIFT_COH | directional drift | ✅ real (gaussian ramp 0.39); blind to non-directional |
| CLUST_DRIFT | unknown — never uniquely separates anything | drop candidate (7→6), pending more error types |

**Rule:** an axis stays only if it has a real ✅ unique contribution OR is the sole candidate
for an unmapped mechanism (load-bearing). CLD is kept *only* because it's the sole type-b
candidate — that claim must be tested, not assumed.

---

## Per-Axis Blind Spot (what each axis is expected NOT to detect)

A measurement system is defined as much by its blind spots as its detections.

| axis | expected blind spot | verified? |
|------|---------------------|-----------|
| DEVIATION | energy-normal corruption / near-OOD (energy decreases) | ✅ (reads ~0 on corruption, correct) |
| CONSENSUS | confident wrong-cluster (type-b: votes don't split) | 🟢 expected; confirm on real type-b |
| CLUSTER_DISTANCE | between-cluster ambiguity if equidistant (small margin) | 🟢 expected (valley geometry) |
| SUBNET_CONSENSUS | type-b (all subspaces agree on the wrong class) | 🟢 expected |
| DRIFT_COH | non-directional drift (no consistent direction) | ✅ (motion_blur ramp 0.09, correct) |
| PERSIST | single transient anomaly (one batch, no streak) | 🟢 expected; confirm |
| CLUST_DRIFT | (unknown role → unknown blind spot) | ⬜ |

---

## Escalation Rules (this is what makes the map a decision engine)

Read the map, then act:

| condition in the map | required action |
|----------------------|-----------------|
| a row has **no ✅/🟢 in any axis** | uncovered mechanism → **propose a new axis** |
| a cell is **🟡 (synthetic) or 🟢 (theory)** for a load-bearing claim | **prioritize a real-data experiment** to promote or refute it |
| **multiple axes always co-fire** on every mechanism (no split) | **redundancy investigation** (within-group + split test); collapse if confirmed |
| **one axis is the sole ✅** in a row | mark **load-bearing**, never drop (CLD for type-b, PER for sustained) |
| an axis is **never uniquely ✅** anywhere | **drop candidate**; confirm across more mechanisms first (CLUST_DRIFT) |
| a mechanism is **UNKNOWN confidence** | it's a measurement gap, not a covered cell — schedule it |

---

## Model Generalization (integrates with the Scale Gate)

The map (rows, columns, mechanisms) stays FIXED across models. Only the *filled cells*
change. Repeat the whole matrix per backbone; divergence between columns is the signal.

| filled for: | ResNet20 | ResNet56 | ViT |
|-------------|----------|----------|-----|
| (current state) | in progress | ⬜ | ⬜ |

Pre-commitment: a detection that holds on ResNet20 but breaks on ResNet56/ViT is a
scale-fragility finding, recorded here — not silently dropped. (Scale-confirmation-bias
guard: "bigger model = better" is a warning, not validation.)

---

## Freeze: the Error-Mechanism taxonomy is closed by default

Do not casually add rows. A new row must satisfy **all four**:
1. represents a genuinely different failure *mechanism* (not a phenomenon/dataset);
2. cannot be expressed as another row **plus severity** (else → Threshold Matrix);
3. has a plausible deployment scenario;
4. requires at least one **distinct** detector (else it's covered).

Otherwise it belongs as a severity level, a parameter, or an appendix — not a mechanism row.

*Example application:* "very weak disturbance" failed criterion 2 (it's severity) → moved to
the Threshold Matrix. "iSUN / far-OOD" is borderline: it's energy-decrease at higher
magnitude (criterion 2 questionable) → currently folded into row 2, not its own row.

---

## Session findings (real-data, this run) — promotions and drops

**Verified facts (✅, promote):**
- **Valleys exist** in penult: separation ratio 2.87 (ResNet20), 4.12 (ResNet56). Distance
  axes are well-founded.
- **Structure axes NOT redundant**: low-severity split (spread 0.32) + distinct detection
  floors. Keep all 7.
- **Recovery → DRIFT_COH_signed** (sep 1.99); recovery reverses direction, signed drift reads it.
- **Directional drift → DRIFT_COH** (sep>1.4 on contrast/gaussian/pixelate).
- **Sustained → PERSIST** (block vs shuffle sep 2.3–5.8).

**Reframed:**
- **type-b sits on RIDGES** (small margin 0.535 vs 0.764), not deep in wrong valleys.
- **VALLEY_MARGIN is the best per-sample type-b detector** (0.895 > CLUSTER_DISTANCE 0.763).
  Promote it as a candidate axis for per-sample type-b.
- **Batch-averaging inflates AUC to a 1.00 ceiling** — hides axis differences and true
  difficulty. Per-sample evaluation is now standard.

**Drops / variants:**
- **TRAJ_LOOP dropped** — PH on 12-pt trajectories too noisy (sep ~0); DRIFT_COH_signed wins.
- **CLUST_DRIFT** still a drop candidate (no unique value).
- **Mahalanobis(CLUSTER_DISTANCE)** — optional variant; helps fog/contrast (+0.11–0.14),
  not elsewhere.

**Open gap (propose new axis):**
- **Row 6 non-directional drift** on brightness/defocus/motion — caught weakly by both
  PERSIST and DRIFT_COH. The clearest new-axis target.

**Scale (ResNet20→56):** picture holds and strengthens; but ResNet56 also has higher accuracy,
so "improves with scale" needs the without-method control. Solid claim: method does not break.

---

## Current honest status (what we actually know vs expect)

- **Verified (✅, real):** energy-decrease→structure axes; directional-drift→DRIFT_COH;
  sustained→PERSIST. Three solid mappings.
- **The decisive gap:** row 4 (type-b). CLD is the sole candidate and is ⬜ on real type-b.
  Everything about the structure-axis column count hinges on it.
- **Expectations not yet facts (🟢):** every type-b blind-spot claim; recovery behavior;
  non-directional-drift coverage by PERSIST. These are reasoning, not measurements.
- **Drop candidate:** CLUST_DRIFT (never uniquely ✅).
- **Synthetic-only (🟡):** energy-explosion (DEV) — no real stimulus exists yet.

Update this document after every real-data run. Promote 🟢/🟡 → ✅ **only** when a real
experiment measured it.
