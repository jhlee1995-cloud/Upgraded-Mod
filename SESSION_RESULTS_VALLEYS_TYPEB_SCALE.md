# Major Real-Data Session — Valleys, type-b, Redundancy, Scale

One merged extraction (8 corruptions × severity 1–3 + type-b + cluster-geometry + ramp +
recovery, real CIFAR-10/10-C) plus two standalone checks (mahalanobis, scale). Seven
findings, several overturning prior assumptions. **Backbone:** ResNet20 penult (+ ResNet56
for scale). All AUC/separation on real data.

---

## Decision → Evidence

| decision | based on (numbers) |
|----------|--------------------|
| **Valleys exist in penult** (distance axes well-founded) | between/within separation ratio **2.87** (worst-case 1.74); STRONG |
| **Clusters are elongated** → mahalanobis justified (partial) | per-class covariance anisotropy **7.4** (top eig / rest) |
| **Mahalanobis is a targeted fix, not a replacement** | helps contrast (0.51→0.65), fog (0.51→0.62); hurts defocus/brightness/gaussian (−0.01 to −0.04); worse on type-b (0.92 vs 1.00) |
| **type-b sits on RIDGES, not in wrong-valleys** | 2D-proj margin: correct 0.764 vs type-b 0.535 vs conf-wrong 0.547 (small margin = ridge) |
| **Batch-averaging inflated type-b to a false ceiling** | batch AUC all 1.00; **per-sample**: CLUSTER 0.763, energy 0.757, **margin 0.895** |
| **valley_margin is the best per-sample type-b detector** | margin 0.895 > CLUSTER_DISTANCE 0.763 per-sample (both models) |
| **Structure axes are NOT redundant (keep 7)** | within-group corr drops 0.99→0.49–0.75 (common mode); at low severity they SPLIT (max spread 0.32); detection floors differ |
| **TRAJ_LOOP (PH) dropped** | recovery-vs-ramp TRAJ_LOOP sep ~0 (0.13 max) — PH on 12-pt trajectories too noisy |
| **Recovery → DRIFT_COH_signed** | recovery vs ramp signed-drift mean sep **1.99** (contrast 3.3, pixelate 3.5) |
| **Method holds at scale** | ResNet20→56: valley 2.87→4.12, type-b margin 0.895→0.905 |

---

## 1. Valleys exist — and the clusters are elongated

Separation ratio 2.87 (mean between-cluster distance / mean within-cluster spread), worst
case 1.74. **The activation space really is shaped like valleys** — distance-based axes are
well-founded, not assuming structure that isn't there.

But the clusters are **elongated** (covariance anisotropy 7.4), so Euclidean `mean_nearest`
distorts. **Mahalanobis** (covariance-corrected distance) was tested: it *helps* exactly where
CLUSTER_DISTANCE was weakest — contrast (0.51→0.65), fog (0.51→0.62) — but slightly *hurts*
elsewhere and is *worse* on type-b. Verdict: a **targeted variant** for low-contrast,
energy-preserving disturbances, not a blanket replacement.

---

## 2. type-b is not what we assumed (the biggest reframing)

Real type-b = CIFAR-10 images the model classifies **confidently and wrongly** (789 wrong,
528 confident-wrong of ~10k). Two surprises:

**(a) type-b sits on RIDGES between clusters, not deep in a wrong valley.** Synthetic had
predicted "deep in the wrong cluster" (margin like clean). Real type-b has a *small* margin
(0.535 vs 0.764 correct) — it's *between* clusters, ambiguous. **Joohoon's "stuck in the
valley between clusters" intuition was geometrically correct for real data.**

**(b) Batch-averaging was inflating detectability to a false ceiling.** At batch level every
axis scored AUC 1.00 — type-b looked trivially easy and the axes looked redundant. **Per
sample** (no averaging) the truth emerges:

| measure | per-sample AUC (correct vs conf-wrong) |
|---------|----------------------------------------|
| **VALLEY_MARGIN** | **0.895** |
| CLUSTER_DISTANCE | 0.763 |
| energy | 0.757 |

**margin beats raw distance** — the "how stuck between clusters" measure is the best
per-sample type-b detector. valley_margin, which looked redundant with SUBNET_CONSENSUS at
the batch ceiling, is **rehabilitated as a per-sample type-b axis**.

*Methodology note (standing):* batch-averaging inflates AUC toward 1.00, hiding both axis
*differences* and true *difficulty*. Per-sample evaluation is required to see either.

---

## 3. Structure axes are NOT redundant (long-open question resolved)

The batch-level +0.96–0.99 correlation among CONSENSUS/CLUSTER_DISTANCE/SUBNET_CONSENSUS was
**clean-corrupt common mode**: within a single group it drops to +0.49–0.75. And decisively,
two independent tests show genuine difference:

- **Low-severity split** (escaping the AUC ceiling): max structure-axis spread **0.32**
  (fog_s2: CONSENSUS 0.90, CLUSTER 0.65, SUBNET 0.97; contrast_s1 spread 0.30).
- **Different detection floors:** SUBNET_CONSENSUS is most sensitive (uniquely catches
  brightness; fog/contrast at severity 1), CLUSTER_DISTANCE least sensitive (fog only at
  severity 3).

Different sensitivity profiles → **keep all four structure-related axes; do not collapse
7→5.** The earlier "redundant" impression was the ceiling artifact.

---

## 4. TRAJ_LOOP failed — and an existing axis does recovery better

The Takens-persistent-homology trajectory axis (built this session for recovery detection)
**failed on real data**: recovery vs ramp separation ~0 (0.13 max). PH on 12-point
trajectories is too noise-sensitive in practice.

But the replacement was already in the roster: **recovery reverses direction at its peak, and
DRIFT_COH_signed reads this.** Recovery vs ramp, signed local-coherence:

| axis | mean sep (recovery vs ramp) |
|------|------------------------------|
| DRIFT_COH_signed | **1.99** (contrast 3.3, pixelate 3.5) |
| CLUST_DRIFT_signed | 1.54 |

So **row 9 (recovery) → DRIFT_COH_signed**, not a new PH axis. A false-redundancy lesson in
reverse: we built a new axis, then found an existing one does the job better. **TRAJ_LOOP is
dropped.**

---

## 5. Temporal map rows filled

| row | mechanism | detector | evidence |
|-----|-----------|----------|----------|
| 5 | directional drift | **DRIFT_COH** ✅ | contrast/gaussian/pixelate sep>1.4; non-directional (brightness/motion/fog/snow) correctly sep~0 |
| 6 | non-directional drift | **PERSIST** (partial) | covers fog/snow (sep 1.0/2.6); **gap**: brightness/defocus/motion (both DRIFT_COH and PERSIST weak) |
| 7 | sustained | **PERSIST** ✅ | block vs shuffle sep 2.3–5.8 (only brightness weak — barely perturbs) |
| 9 | recovery | **DRIFT_COH_signed** ✅ | recovery vs ramp sep 1.99 |

**Row 6 is the clearest "propose a new axis" candidate** — non-directional gradual drift on
brightness/defocus/motion is caught weakly by everything we have.

---

## 6. Scale generalization (ResNet20 → ResNet56)

Same measurements, larger model, same data:

| metric | ResNet20 | ResNet56 |
|--------|----------|----------|
| valley separation | 2.87 | **4.12** |
| per-sample type-b margin AUC | 0.895 | **0.905** |
| per-sample type-b CLUSTER AUC | 0.763 | **0.886** |

**The picture holds and strengthens at scale** — cleaner valleys, more detectable type-b.

*Scale-confirmation-bias guard:* ResNet56 also has higher accuracy (0.943 vs 0.921) and fewer
errors, so "better" may be model quality, not method merit. The **solid** claim is *the method
does not break at scale*; "improves with scale" needs the without-method control before it's a
fact.

---

## Axis roster after this session

| axis | status | reason |
|------|--------|--------|
| DEVIATION | **keep** | distinct sensitivity (contrast s1); energy explosion (synthetic) |
| CONSENSUS | **keep** | vote-split; distinct floor |
| CLUSTER_DISTANCE | **keep** | distinct (least sensitive but real); mahalanobis variant optional |
| SUBNET_CONSENSUS | **keep** | most sensitive; uniquely catches brightness |
| DRIFT_COH | **keep** | directional drift *and* recovery (signed) |
| PERSIST | **keep** | sustained (strong); partial non-directional |
| CLUST_DRIFT | **drop candidate** | no unique value across any mechanism |
| TRAJ_LOOP | **drop** | DRIFT_COH_signed does recovery better |
| **VALLEY_MARGIN** | **promote (candidate)** | best per-sample type-b detector (0.895) |
| mahalanobis(CLD) | optional variant | fixes fog/contrast only |

---

## Throughline

This session overturned three assumptions, each caught by looking past a ceiling or a
synthetic prediction:
1. "type-b is deep in wrong valleys" → it's on **ridges** (margin small); margin detects it.
2. "structure axes are redundant" → ceiling artifact; at low severity they **split**.
3. "we need a PH axis for recovery" → **signed drift** already does it better.

And one intuition was vindicated: **valley_margin — "how stuck between clusters" — is the best
per-sample type-b detector.** The recurring method that made these visible: escape the
batch/severity ceiling, and check signed/geometric measures the standard axes don't encode.

---

## Next

- **Row 6 new-axis candidate** — design a detector for non-directional gradual drift
  (brightness/defocus/motion) that PERSIST+DRIFT_COH miss.
- **valley_margin** — add as a permanent per-sample axis; re-run the coverage map with it.
- **Scale control** — the "improves with scale" claim needs the without-method baseline.
- **Per-sample evaluation** — adopt as standard alongside batch (batch hides difficulty).
