# Vision — Conditional Long-Term Picture

**Deliberately separate from `MASTER_SUMMARY.md`.** The master holds verified facts and the
immediate plan. This holds the *conditional* picture — what the project *could* become *if*
certain hypotheses hold. Separation prevents vision from hardening into assumed fact (the
failure mode the Evidence Status table guards against). Everything here is prefixed by **IF**.

---

## The objective never changes

**Deployment is the objective. Publications are optional.** Success metric:

1. The measurement layer consistently improves real systems.
2. The measurement generalizes across architectures.
3. The controller safely changes model behavior.
4. A deployed robot makes better decisions because of these measurements.

A geometry-based measurement science would be a *by-product* of building a deployable
controller, never a replacement. If at any point "the science" pulls effort away from a working
robot controller, that is a warning, not progress.

---

## The conditional question

The center of gravity has shifted from the original framing —

> Can we build a label-free, gradient-free test-time-adaptation controller?

toward a deeper question the data surfaced —

> Do confident-but-wrong (type-b) samples consistently sit in low-margin ridge regions between
> competing semantic clusters, across fundamentally different representation learners?

This is real and interesting, but it is **one ViT experiment away** from being supported or
refuted, and the structure below rests entirely on it.

---

## IF margin is architecture-independent (Hypothesis A)

*If* margin survives ResNet -> ViT -> (DINO, CLIP), flagging type-b as a low-margin ridge
regardless of how the representation was learned, then:

- Gate 1 is not a CNN detector but a **general measurement principle for representation geometry**.
- The project contributes to **representation learning**, not only adaptation: it evaluates a
  representation by its *internal geometry* (does it organize ambiguity into ridges/valleys?)
  rather than by outcome metrics (accuracy, linear-probe, retrieval).
- `correct -> deep valley -> large margin` / `confident-wrong -> ridge -> small margin` would be
  a property of the *representation itself*, not the learning algorithm.
- Adaptation becomes the **first application** of a broader measurement framework, potentially
  reusable across robotics, autonomous driving, and multimodal systems.

This expansive picture is **conditional on Hypothesis A**, currently untested beyond two CNNs.

## IF margin is CNN-specific (Hypothesis B)

*If* margin disappears outside CNNs, it is still a useful signal for the (likely CNN) deployment
model, and we have learned something real: convolutional and transformer representations
organize semantic ambiguity *differently*. The deployment path is unaffected (the robot model
is probably a CNN); the grander measurement-science vision simply does not open.

**Both outcomes keep the deployment goal intact.** A widens the scientific contribution, B
narrows it. Neither changes whether we can mount a controller on a robot.

---

## Phased roadmap

- **Phase 1 — Measurement basis. Complete.** Axis discovery, real-data validation, error
  taxonomy, three-gate architecture.
- **Phase 2 — Representation geometry. Current.** Verify margin on ViT (the one decisive test).
  Exit criteria: margin > absolute distance for type-b; type-b stays a low-margin phenomenon.
  DINO/CLIP only if ViT survives.
- **Phase 3 — Closed-loop control.** *Only after Phase 2.* Integrate Gate 1 into a real
  adaptation controller; measure prevented bad adaptation, safer decisions, improved robustness.
- **Phase 4 — Robotics.** Mount on an open-source robot; measure adapt/hold, escalation,
  recovery, real-world safety behavior. **This is the actual bar.**

The long-term aim, *if the geometry generalizes*, is a **measurement framework for neural-network
internal representations** — shifting the question from "how should a model learn?" to "how can
we measure whether the model's representation should be trusted?" But that is the conditional
prize; the unconditional objective is the deployed controller.
