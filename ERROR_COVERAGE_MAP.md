# Error-Coverage Map — Master Tracking Document

**Goal:** fill every cell. Each row is an *error mechanism* (not a dataset — mechanism,
because that's what determines which axis fires). Each column is a *detection axis*. Each
cell records whether that axis catches that error, with the evidence.

**Legend:**
- ✅ = caught, real-data evidence
- 🟡 = caught on synthetic only (real test pending)
- ⬜ = no data / not tested yet
- ✗ = tested, does NOT catch (and that's understood/correct)
- — = not applicable

**Axes (current pool of 7):**
DEV=DEVIATION (energy inflation, one-sided) · CON=CONSENSUS (hard-vote disagreement) ·
CLD=CLUSTER_DISTANCE (nearest-center distance) · SUB=SUBNET_CONSENSUS (soft entropy) ·
DCH=DRIFT_COH (directional drift) · PER=PERSIST (sustained) · CDR=CLUST_DRIFT (distance-vector drift)

---

## The Map

| # | error mechanism | example | DEV | CON | CLD | SUB | DCH | PER | CDR | test status |
|---|-----------------|---------|-----|-----|-----|-----|-----|-----|-----|-------------|
| 1 | **energy explosion** | sensor glitch, NaN-ish spike | 🟡 | — | — | — | — | — | — | synthetic only; no real stimulus |
| 2 | **energy decrease** | corruption, near-OOD at penult | ✗ | ✅ | ✅ | ✅ | — | — | — | real (CIFAR-10-C, CIFAR-100) |
| 3 | **between-cluster (type-a)** | ambiguous input | ✗ | ✅ | 🟡 | ✅ | — | — | — | real corruption ✅; clean type-a ⬜ |
| 4 | **wrong-cluster (type-b)** | confident-wrong, energy-normal | ✗ | ✗ | ⬜ | ✗ | — | — | — | **NEEDS iSUN** — biggest gap |
| 5 | **directional gradual drift** | sensor aging, consistent shift | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | 🟡→drop | real (gaussian ramp 0.39); CDR covered |
| 6 | **non-directional gradual drift** | blur increasing | ⬜ | ⬜ | ⬜ | ⬜ | ✗ | ✅ | ⬜ | DCH blind (motion 0.09); PER? real-check ⬜ |
| 7 | **sustained disturbance** | persistent weather/occlusion | ⬜ | ⬜ | ⬜ | ⬜ | — | ✅ | ✗ | real (block 0.51 vs shuffle 0.03) |
| 8 | **adversarial** | FGSM/PGD attack | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | **no data, never tested** |
| 9 | **recovery** | drift returning to normal | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | **never tested** (matters for adapt-stop) |
| 10 | **very weak disturbance** | severity-1 corruption | ⬜ | ⬜ | ⬜ | ⬜ | — | — | — | only severity-3 tested; detection floor ⬜ |

---

## What's filled, what's not

**Solidly mapped (real-data ✅):**
- **energy decrease** (corruption/near-OOD) → structure axes (CON/CLD/SUB), confirmed.
- **directional drift** → DCH (gaussian ramp 0.39, solid).
- **sustained** → PER (block 0.51 vs shuffle 0.03, rock-solid).

**The empty cells (⬜) — the actual question "what can't we map yet":**

1. **type-b (row 4)** — THE biggest gap. CLD is the only candidate detector and it's
   *untested on real type-b*. Synthetic always made type-b energy-large (so DEV/CLD both
   fired = false redundancy). Needs **iSUN** (energy-normal near-OOD for CIFAR-100). Until
   this runs, we don't know if CLD actually catches real confident-wrong inputs.

2. **non-directional drift (row 6)** — DCH is blind by design (correct). Open question: does
   PER cover it on real data? If yes, row 6 is fine; if no, it's a real gap needing a new axis.

3. **adversarial (row 8)** — entirely empty. Different mechanism (tiny perturbation, big
   misclassification). No data (need FGSM/PGD), never tested. May need a new axis.

4. **recovery (row 9)** — entirely empty. Do PER/DCH distinguish "getting worse" from
   "getting better"? Critical for the adaptation decision (stop adapting if recovering).

5. **energy explosion (row 1)** — synthetic-only. DEV is the designed detector but there's
   no real stimulus to confirm.

6. **weak disturbance (row 10)** — only severity-3 tested. Where's the detection floor?
   Does the map hold at severity-1?

**Cross-cutting unknown (affects whether columns collapse):**
- Are CON/CLD/SUB really 3 axes or 1? **PARTIALLY RESOLVED (run7):** in mixed streams they
  hit +0.96–0.99, but the common-mode test (within-group correlation) drops this to **+0.60–0.67**
  — so most of the 0.99 was clean-corrupt common mode, *not* genuine redundancy. However 0.6
  is still moderate, and on every tested disturbance (corruption, CIFAR-100) all 3 structure
  axes fire identically (AUC 1.00, split ≤0.15) — they look redundant *on these errors*.
  **BUT the verdict is not final:** the tested errors are all type-a-like (between-cluster),
  where all 3 co-fire. The decisive stimulus is **type-b** (wrong-cluster, energy-normal),
  where CONSENSUS goes silent (votes don't split) but CLUSTER_DISTANCE alone fires — the axes
  would *split* there. If they split on type-b → keep all 3 (CLD is type-b's unique detector,
  dropping it = memory#18 false-redundancy trap). If they don't → genuinely redundant, 7→5.
  **This is why iSUN is decisive: it fills row 4 AND decides the column count.**

---

## Roadmap to fill the map (priority order)

1. **iSUN → row 4 (type-b)** + resolve CLD ⊥ DEV. The single highest-value gap. Data:
   HuggingFace `detectors/isun-ood` (8,925 energy-normal natural scenes).
2. **structure_redundancy.py → columns 2–4.** Decide if structure axes are 3 or 1 (already
   built; needs run with `--diag`). Determines the map's column count.
3. **PER on non-directional drift → row 6.** Does PER cover what DCH misses? (ramp on
   motion_blur, already have data — just analyze PER there.)
4. **severity sweep → row 10.** Re-run streams at severity 1,2,3,4,5; find detection floor.
5. **recovery streams → row 9.** Build a stream that ramps severity UP then DOWN; check if
   any axis reads the direction.
6. **adversarial → row 8.** Generate FGSM/PGD on CIFAR-10; test all axes. (New data
   generation; lower priority — different threat model from the deployment target.)

**After the map is full:** the empty cells that *remain* after data is available are where
**new axis candidates** are needed. That's the entry point to systematic add/remove axis
exploration — a cell no existing axis fills is a new-axis requirement, made precise.

---

## How to read this for add/remove decisions

- A **column that's ✗/⬜ everywhere except where another column is also ✅** → redundant axis,
  drop candidate (this is how CLUST_DRIFT became a drop candidate: never the unique ✅).
- A **row with no ✅ in any column** → uncovered error → new axis needed.
- A **column that's the *only* ✅ in some row** → load-bearing axis, never drop (e.g. CLD is
  the only candidate for row 4; PER is the only ✅ for row 7).

This map is the single source of truth for "which axes matter." Update it after every
real-data run.
