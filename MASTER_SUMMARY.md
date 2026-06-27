# Label-Free Modulator — Master Summary

The single document for project state. A label-free, gradient-free control-plane modulator
for neural-network test-time adaptation, targeting robotics / autonomous-vehicle perception.
Reads only activation statistics + predicted-class distributions (no loss, gradients, or
labels) to decide **adapt / hold / escalate**. Validated on real CIFAR-10/10-C and ImageNet.

---

## The three-gate architecture (the system's decision structure)

The system is not one detector but **three gates at different signal sources and timescales**:

| gate | signal source | timescale | decides | mechanism |
|------|---------------|-----------|---------|-----------|
| **1. adapt/hold** | penult activation geometry (7 axes) | per-batch (fast) | trust model's topology vs current data | distance / margin / vote / energy / drift |
| **2. action** | (downstream, project-specific) | per-decision | act vs fall back | — |
| **3. escalate** | predicted-class **distribution over time** | time-window (slow) | "outside AI competence → call a human" | **density acceleration** |

Gates 1 and 3 are **fundamentally different signal sources**: gate 1 reads where individual
points sit in the activation landscape; gate 3 reads how the *set-level density* over that
landscape changes through time. A point-wise axis cannot see a distribution shift, and a
distribution monitor cannot see a single wrong point — they are complementary by construction.

---

## Gate 1 — the activation-geometry axes (adapt/hold)

### What is measured
**Model-topology ↔ data misalignment.** Clean data forms a multi-center constellation
(per-class manifolds = "valleys") frozen at train time, acting as a coordinate frame. An axis
asks: has this input drifted off that frame, and how?

### The roster (after CIFAR + ImageNet validation)

| axis | what it catches | fate | evidence |
|------|-----------------|------|----------|
| **DEVIATION** | energy explosion (glitch) — one-sided | **keep** | tightest safety margin; distinct floor (contrast s1) |
| **CONSENSUS** | structure-broken nonsense (vote split) | **keep** | distinct sensitivity floor |
| **CLUSTER_DISTANCE** | off-manifold distance | **keep, but scale-fragile** | degrades when valleys shallow (ImageNet 0.636) |
| **SUBNET_CONSENSUS** | structure break (parallel subnets) | **keep** | most sensitive; uniquely catches brightness |
| **MARGIN** (valley_margin) | type-b: stuck between two clusters | **PROMOTE to core** | best type-b detector, scale-robust (0.895 / 0.800) |
| **DRIFT_COH** | directional drift + recovery (signed) | **keep** | drift sep>1.4; recovery sep 1.99 |
| **PERSIST** | sustained anomaly (streak) | **keep** | block-vs-shuffle sep 2.3–5.8 |
| CLUST_DRIFT | (nothing unique) | **drop candidate** | no unique coverage any mechanism |
| TRAJ_LOOP (Takens-PH) | (recovery — but worse) | **dropped** | PH on 12-pt traj too noisy; DRIFT_COH_signed wins |
| mahalanobis(CLD) | fog/contrast specifically | **optional variant** | +0.11–0.14 on fog/contrast only |

**Near-OOD is two structurally distinct types needing complementary signals:**
- **type-a** (between clusters): votes split → CONSENSUS/SUBNET catch it.
- **type-b** (toward one wrong center, confident-wrong): votes do NOT split (all subspaces
  agree on the wrong class) → CONSENSUS is structurally blind; only **distance/margin** sees
  it. Real type-b sits on **ridges between clusters** (small margin), not deep in a wrong
  valley. **MARGIN is the detector.**

### Why MARGIN is the load-bearing type-b axis (the ImageNet lesson)
CIFAR alone said "valleys deep (sep 2.87), all distance axes good." ImageNet overturned half
of that: at 1000 classes in 2048d, **valleys are shallow (sep 1.14)** — clusters overlap
(120 dog breeds nearly touching). Absolute **CLUSTER_DISTANCE degrades** (0.763→0.636) because
"distance to nearest center" loses meaning when centers overlap. But **MARGIN survives**
(0.895→0.800) because it measures the *ambiguity between the top-2 clusters*, not absolute
depth — and that is exactly what type-b is, at any scale. **Margin doesn't depend on valley
depth.** This is the single most important scale finding: the absolute-distance picture is
scale-fragile; the relative-margin picture is scale-robust.

**Remaining risk:** only CNNs tested (ResNet20, ResNet50). Whether margin is a
*representation-independent* geometric phenomenon or a CNN-specific artifact is the project's
single biggest open question — it must be tested on ViT, then self-supervised (DINO) and
multimodal (CLIP) encoders before margin can be called universally load-bearing. Until then,
margin is a **core candidate**, not a settled core axis.

### Detection floors differ → axes are NOT redundant (keep them all)
The batch-level 0.99 correlation was clean-corrupt **common mode** (within-group 0.49–0.75).
At low severity (below the AUC ceiling) the structure axes **split** (max spread 0.32). And
per-axis detection floors differ: SUBNET_CONSENSUS most sensitive (catches brightness, fog/
contrast at s1), CLUSTER_DISTANCE least (fog only at s3). Different sensitivity profiles =
do not collapse the roster.

---

## Gate 3 — density acceleration (escalate to human)

### The blind spot that motivated it
Stress-testing the axes over a *range* (sweep, not single point) found:
- **Adversarial (FGSM)**: caught at every epsilon (batch AUC 1.00). Not a blind spot.
- **Partial corruption**: caught down to 10% corrupted area. Not a blind spot.
- **Distribution shift (class imbalance)**: **all 7 axes flat at ~0.5 across the full skew
  range.** A single-class batch is invisible. **Structural blind spot.**

### Why it's a blind spot — and why that's by design
In topology terms: the **landscape** (cluster positions/shapes, frozen at train) is unchanged;
only the **measure** (how many points land in which valley) shifts. Every individual point is
perfectly aligned (correct cluster, deep margin) — the axes, being point-wise, see nothing.
Distribution shift is a **measure** change, not a **topology** change. The axes are designed to
catch topology misalignment, so missing a measure change is correct, not a bug.

### The reframe (decisive)
Distribution shift is **not a classification error** — the model classifies every point
correctly, so for trust/hold it's harmless. The danger is (a) adapting on skewed input biases
the model, or (b) it signals a **situation outside AI competence**. And the right signal is
**not** a comparison to stored train density (deployment distribution legitimately differs from
train → false alarms) but the **acceleration (2nd derivative)** of the distribution's change:
gradual drift (day→night) is normal = low acceleration; an **abrupt** change (emergency
vehicles suddenly 10×) is the signal = high acceleration → a human should assess the situation.

### Validated mechanism (CIFAR)
Three streams — stable / gradual-drift / sudden-jump:

| stream | velocity (1st diff) | **peak acceleration (2nd diff)** |
|--------|--------------------|-----------------------------------|
| stable | 0.295 | 0.170 |
| gradual | 0.295 | 0.191 |
| **sudden** | 0.305 | **0.658** (3.4×) |

**Acceleration separates sudden (3.4×); gradual stays quiet.** Critically, **velocity alone
fails** (all ~0.30) — batch-sampling noise floods the 1st derivative, and only the 2nd
difference cancels that noise, leaving the genuine abrupt change. So acceleration is
*necessary*, not just nicer.

### Validated meaning (ImageNet) + the two-layer design
Spiking a class to 40% of the batch at t≥10, peak acceleration:

| scenario | peak accel |
|----------|-----------|
| stable (no spike) | 0.141 |
| **ambulance** (407) | **0.859** |
| fire_engine (555) | 0.719 |
| police_van (734) | 0.766 |
| tabby_cat (281) | 0.672 |
| golden_retriever (207) | 0.688 |
| coffee_mug (504) | 0.703 |

Every abrupt spike is **5–6× the stable baseline** — the mechanism holds with real safety-
relevant classes. **Critical insight: critical and mundane spikes give similar acceleration
(0.781 vs 0.688).** This is correct and important: the gate detects *that an abrupt shift
happened*, it does not read class identity. So **escalate is a two-layer design**:
- **Layer 1 — density acceleration:** class-agnostic abruptness detector. Fires on any abrupt
  distribution change (cat surge and ambulance surge alike).
- **Layer 2 — policy map:** which classes are safety-critical enough that their abrupt spike
  warrants a human (ambulance → escalate; cat → log).

A cat surge *is* an abrupt shift (same accel) but harmless; an ambulance surge is the same-size
shift but dangerous — the difference is *which class*, decided by policy, not by accel
magnitude. Clean separation of concerns: **detection (gate) vs judgment (policy).** The gate is
now validated at all three levels — mechanism (CIFAR), meaning (ImageNet 6×), architecture
(two-layer).

### Open in this thread
- Threshold sweep for the escalate trigger (where to cut accel).
- Define the policy map (layer 2) — which ImageNet/deployment classes are escalate-worthy.

---

## Methodology principles (earned the hard way)

1. **Escape every ceiling.** Three ceilings hid the truth: synthetic AUC 1.00, severity AUC
   1.00, and **batch-averaging** AUC 1.00. Batch-mean inflates type-b detectability to a false
   ceiling — per-sample (no averaging) reveals both axis differences and true difficulty.
   *Per-sample evaluation is now standard alongside batch.*

2. **Never trust a redundancy verdict from a ceiling.** Co-firing at a ceiling = "the test
   failed to separate them," not "they are redundant." Confirm redundancy only below the
   ceiling with the specific separating input. (Structure axes looked redundant at the
   ceiling; split at low severity.)

3. **Validate the instrument before the result.** Three measurement bugs this session (stream-
   builder alternating-vs-sustained, recovery-triangle skipping levels, two `def`-header
   clobbers from str_replace) were caught by *eyeballing the pattern* and testing
   `import + callable(main)`, not just syntax. A worry must be verified before it's recorded;
   a verified-and-retracted worry strengthens the asset it threatened.

4. **Sweep, don't pick.** Treat the sweep curve's *shape* as the measurement: step = discrete
   threshold, gentle slope = weak continuous, flat = blind spot. Seed-stability of the curve
   is the meta-validation.

5. **Scale-confirmation-bias guard.** Bigger/cleaner models make any method look better;
   "works at scale" is a warning, not validation. The deployment number is small-model
   performance; "improves with scale" needs the without-method control. (ResNet56 looked
   better but also had higher accuracy — confounded.)

6. **Real data overturns synthetic intuition repeatedly.** Energy *decreases* under real
   corruption (not increases); type-b sits on *ridges* (not in wrong valleys); recovery is a
   *folded segment* not a loop (naive PH failed); valleys are *shallow* at ImageNet scale.
   Synthetic is for instrument-checking; real data decides.

---

## Scale generalization summary (CIFAR → ImageNet)

| property | CIFAR-10 (ResNet20, 64d) | ImageNet (ResNet50, 2048d) | verdict |
|----------|--------------------------|----------------------------|---------|
| classes | 10 (semantically far) | 1000 (many adjacent) | — |
| accuracy | 0.921 | 0.771 | — |
| valley separation | 2.87 (deep) | **1.14 (shallow)** | absolute-distance picture is scale-fragile |
| type-b count | 789 | 2300 | ImageNet richer |
| MARGIN type-b AUC | 0.895 | **0.800** | **scale-robust — margin is load-bearing** |
| CLUSTER_DIST type-b | 0.763 | 0.636 | degrades with shallow valleys |
| density-accel mechanism | validated (3.4×) | semantic test in progress | — |

---

## Evidence Status (the project's truth table)

Separating what is *verified* from what is *hypothesis*, so claims don't silently harden into
assumed facts. ✅ verified · 🟡 mechanism validated, scope-limited · ⏳ not yet tested · ❌ ruled out.

| claim | status | scope of evidence |
|-------|--------|-------------------|
| Valleys exist (deep) on CIFAR | ✅ | ResNet20, sep 2.87 |
| Valleys shallow at ImageNet scale | ✅ | ResNet50, sep 1.14, not a sample artifact |
| **Margin is the best type-b detector** | ✅ | CIFAR 0.895, ImageNet 0.800 (both CNN) |
| **Margin robust across scale** | ✅ | ResNet20 → ResNet50 |
| **Margin robust across architectures** | ⏳ | *only CNN tested — ViT/DINO/CLIP open* |
| Margin is a representation-independent geometric phenomenon | ⏳ | the biggest open question |
| CLUSTER_DISTANCE is scale-fragile | ✅ | degrades 0.763→0.636 with shallow valleys |
| Structure axes not redundant | ✅ | low-severity split 0.32, distinct floors (CIFAR) |
| Structure-axis behavior at ImageNet scale | ⏳ | not re-run with shallow valleys |
| type-b sits on ridges (small margin) | ✅ | CIFAR 2D-proj; ImageNet consistent |
| Recovery → DRIFT_COH_signed (not PH) | ✅ | sep 1.99 vs TRAJ_LOOP ~0 (CIFAR) |
| TRAJ_LOOP dropped | ✅ | PH on 12-pt traj too noisy |
| Distribution shift is a structural blind spot | ✅ | all 7 axes flat ~0.5 across skew |
| Density acceleration detects abrupt shift | ✅ | CIFAR 3.4×, ImageNet 6× |
| Density acceleration is class-agnostic | ✅ | critical 0.781 ≈ mundane 0.688 |
| Velocity alone is insufficient (accel necessary) | ✅ | velocity ~equal across streams |
| Two-layer escalate (detect + policy) | 🟡 | mechanism validated; policy map undefined |
| Three-gate architecture | 🟡 | internal validation complete; not deployed |
| Row-6 gap (non-directional drift) needs new axis | 🟡 | both PERSIST and DRIFT_COH weak — candidate |
| "Improves with scale" | ⏳ | confounded by model accuracy — needs without-method control |
| Closed-loop adapt/hold control | ⏳ | gates designed, not wired to an adaptation loop |

---

## Open questions / next (priority order)

**P1 — Verify margin on ViT (foundation check, one decisive experiment).** Everything
downstream (closed-loop control -> robot mount) rests on margin being a real signal, not a
CNN-specific artifact. So far it is verified only on CNNs (ResNet20 -> ResNet50). **ViT is the
single decisive test**: it is architecturally fundamentally different (attention, not
convolution), so if margin still flags type-b on ViT, "CNN-specific" is already refuted -- one
experiment settles it. *Exit criteria:* margin > absolute distance for type-b; type-b remains a
low-margin (ridge) phenomenon. *Two outcomes, both useful:* (A) margin survives -> it is a
representation-level property, foundation solid, proceed to closed-loop; (B) margin disappears
outside CNNs -> it is CNN-specific, still usable for the (likely CNN) deployment model, and we
now know the geometry is architecture-dependent. DINO/CLIP/robotics-encoder are **contingent
follow-ups**, only worth running if ViT survives (DINO is itself a ViT backbone -> confirmation,
not a second decisive test). Do NOT pre-plan all four -- run ViT, then decide. *Scope
discipline:* this is foundation-verification for the deployment goal, NOT a pivot into
representation-learning research; the north star remains the robot controller. (See VISION.md
for the conditional longer-term picture, kept separate so it does not harden into assumed fact.)

**P2 — Complete gate 3 (escalate).** Mechanism is validated; finish the policy layer. Threshold
sweep for density acceleration (where to cut); multi-class and mixed-event spikes; define the
safety policy map (which classes/events warrant human escalation). Detection ("something changed
abruptly") is done; policy ("does this require escalation?") is pending.

**P3 — Close the row-6 blind spot.** Non-directional gradual drift (brightness/defocus/motion)
is caught weakly by both PERSIST and DRIFT_COH -- the strongest candidate for an additional
axis. Also: re-run the 7 axes / redundancy at ImageNet scale (shallow valleys may change which
structure axes matter when clusters overlap).

**P4 — Representation-geometry study (theory, secondary).** Useful supporting experiments:
valley depth vs model scale; margin distribution vs class overlap; valley evolution during
training; margin across architectures. Secondary to the deployment path -- pursue only as it
serves P1/P3, not as an end in itself.

**P5 — Formalize the three-gate architecture.** A design document: Gate 1 (activation geometry
-> adapt/hold), Gate 2 (task action -> execute/fallback), Gate 3 (distribution dynamics ->
escalate), Policy Layer (risk weighting), with type->response routing (directional-drift->ADAPT,
magnitude-explosion->HOLD+rollback, type-b->HOLD, abrupt-density->ESCALATE).

**Phase 3 (after P1 succeeds) -- closed-loop control.** Wire the gates to an actual test-time-
adaptation loop (e.g. BN-statistic update) so detection drives real adapt/hold/escalate, and
measure whether the gate prevents bad adaptation (adapting on type-b or skewed input). This is
the step toward robot mount; it follows P1 because it rests on margin being real.

**Phase 4 -- robotics.** Mount the controller on an open-source robot; measure adapt/hold,
escalation, recovery, real-world safety behavior.

Also open: **CLUST_DRIFT final call** (confirm drop or find a disturbance it uniquely catches),
and the **"improves with scale" control** (ResNet56 looked better but had higher accuracy ->
needs a without-method baseline).

---

## Success metric (what "done" means)

The project is successful **not** when it produces a paper. It is successful when:

1. The measurement layer consistently improves real systems.
2. The measurement generalizes across architectures.
3. The controller safely changes model behavior.
4. A deployed robot makes better decisions because of these measurements.

**Publications are optional. Deployment is the objective.** Every phase above is judged against
this: P1 (ViT) is not "is this publishable?" but "is margin real enough to build a controller
on?"; P4 is the actual bar. Scientific results (e.g. margin as an architecture-independent
property) are welcome by-products, never the goal -- they must not pull the project off the
deployment path.

---

## Infrastructure

- **Repo:** `jhlee1995-cloud/Upgraded-Mod` (public, flat). RunPod RTX 4090, persistent volume
  at `/workspace` (now petabyte-scale mount — storage no longer a constraint).
- **Models:** `chenyaofo/pytorch-cifar-models` (resnet20/56, penult 64d); torchvision
  `resnet50` IMAGENET1K_V2 (penult = avgpool, 2048d).
- **Data:** CIFAR-10/10-C (Zenodo tar), ImageNet val (non-gated HF mirror
  `benjamin-paine/imagenet-1k-256x256`, parquet, downloaded via `huggingface_hub` directly —
  NOT the `datasets` library, which crashes the env via pandas/pyarrow upgrade).
- **Caches:** `/workspace/cache/run11` (CIFAR merged), `/workspace/cache/imagenet/penult_N.npz`.
