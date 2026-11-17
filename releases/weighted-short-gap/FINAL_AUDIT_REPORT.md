# Current model — completed opened-match audit

16 November 2026. **Technical verification PASS; overall CONDITIONALLY READY.** The fixed current model, weighted_short_gap, improves on all 18previously opened PFF test matches in both natural and fixed blind scenarios. This is evidence beyond the reused development set, but not a fresh prospective holdout: earlier parent outcomes and weaknesses were known. No weights, thresholds, features or reference definitions were changed during this comparison. No organizer submission occurred.

## Quality evidence

| Cohort | Parent natural Spell F1 | Current natural Spell F1 | Wins | Parent blind | Current blind | Wins |
|---|---:|---:|---:|---:|---:|---:|
| bridge6 | 0.613896 | 0.638797 | 6/6 | 0.594795 | 0.616778 | 6/6 |
| original_final12 | 0.604159 | 0.639452 | 12/12 | 0.586834 | 0.621332 | 12/12 |
| all 18 | 0.607404 | 0.639234 | 18/18 | 0.589488 | 0.619814 | 18/18 |

Current-model PFF training mean on all 36 training games is **0.652584**; reused ten-game development mean **0.680009**; opened18-game mean **0.639234**. Training-minus-opened mean is **+0.013350**. All groups use the same current model and canonical event-proxy reference policy. Different fixtures, shared provider/countries and repeated selection mean this gap is not a proof against overfitting.

All 144 match/scenario/model/reference records are present; canonical and SDK-shadow metrics agree exactly. Every predeclared review guard passes. New evidence and per-match results are in [the complete report](../docs/experiments/recovery-opened-audit/REPORT.md). The parent was recomputed with current audited preprocessing, rather than substituting historical12-game numbers. Small historical-source differences are recorded.

Natural short-spell recall (<3 s,IoU0.5) on the 18 matches improves 123/1216→173/1216 (10.12%→14.23%). Missing-ball false stoppages decrease126855/579297→109345/579297; missed actual stops decrease40504/1070662→27983/1070662. Wrong-team errors during gaps increase62047→64393 frames; the aggregate gain does not erase this tradeoff.

Netherlands–USA improves 0.504449→0.548485, but still predicts 219 possessions against 309references. Only6/76 spells shorter than3 s match atIoU0.5; none of ten sub-second spells match. Among161misses atIoU0.5,33are mostly predicted dead,62 mostly opponent,66 mostly correct-team with bad extent or one-to-one matching competition. First-half Spell slightly regresses; second-half improves 0.437762→0.518310. This is a real residual model/reference-semantics limitation, not a claimed solved case.

## Acceptance gates

| Gate | Status | Evidence and limitation |
|---|---|---|
| Rules and data use | CONDITIONAL overall | No prohibited target-label fitting/scoring in this run. Original PFF access terms, team eligibility and organizer source access remain prior qualifications. |
| Alignment | PASS for tested inputs |18 source identities/hashes, independent period clocks, frame mapping and reference coverage checked; no label-based shift tuning. |
| Official metric | PASS | Pinned official scorer, both reference reconstructions, complete comparison grid. |
| Temporal and input contracts | PASS |140 tests, zero skips/failures/errors, plus60 brute-force decoder comparisons. |
| Submission format | PASS | Exact unchanged CSV and source/weight gate pass official validator. |
| Reproducibility | PASS in tested environments | Prior4 fresh-source macOS/Linux generations remain valid; inference/model/CSV bytes unchanged. Paired audit runtime exactly equals standalone current/parent probabilities on 10502 in both scenarios. One physical host. |
| Evidence quality | PASS with limits | Training/development/reopened-test results separated;18reopened fixtures are not represented as untouched. No hidden score, no-overfit or finalist guarantee. |

## Leakage and reporting repair

The 36/10/12/6 PFF splits have empty match-ID intersections; source receipts contain64distinct tracking hashes. Current65-game neural and 36-game activity registries match approved training identities. The normalizer is exactly the parent train-only normalizer. Geometry/reference separation, estimated-coordinate exclusion, label-invariance and period tests pass. Exhaustive semantic near-duplicate detection across all providers is not established; no claim of zero conceivable leakage is made.

The new audit reporter initially lacked period metadata on augmented observations. This raised a KeyError after predictions and references were saved. It did not affect inference, which received the correct original period spans. The reporter was repaired and regression-tested; unchanged arrays were used to complete all scores. Original failure, frozen source hashes and a narrowly scoped amendment remain recorded. No P0/P1 inference defect is known in the audited scope.

## Artifacts and decision

Retain the already selected weighted_short_gap model. All 18 fixed comparisons and original review guards pass; no heldout-driven parameter search was performed. Training and audit outputs are immutable evidence, not new training input. Short-possession recognition, ambiguous ownership, long gaps and provider/reference shifts remain weaknesses. A future truly new provider-diverse labeled sample would strengthen generalization evidence; it is not replaced by unit-test success.

Current package: out/recovery-ready/. CSV SHA-256:

`1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`

The manifest and technical document remain the exact generation-bound versions. This later report adds validation evidence without altering them. Source branch: release/short-possession-recovery. Public source access and organizer submission remain on hold. Historical audits are preserved in PARENT_FINAL_AUDIT_REPORT.md and DEVELOPMENT_AUDIT_REPORT.md.
