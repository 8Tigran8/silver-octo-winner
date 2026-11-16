# Independent audit — final verdict

**CONDITIONALLY READY**, 16 November 2026. The audited output is reproducible and format-valid. Confirmed implementation defects were repaired; no protected-target label use was found in the inspected code and evidence. The private repository, source-access qualifications, unverified team eligibility and limited fresh generalization evidence prevent an unconditional READY claim.

**Later same-day real-match validation completed:** six previously unopened PFF bridge fixtures now provide prospective evidence for the corrected source. Current Spell F1 **0.613896**, v4 **0.546054**, gains on all six. All 36 PFF training fixtures score **0.625242** under the same frozen current inference, a training-minus-new gap of **0.011345**. This supersedes the earlier assertion that corrected-source evidence was development-only. Partial country overlap, shared provider and proxy reference definitions remain explicit. No fitting or selection used the six outcomes. See [the new-match audit](../docs/experiments/unseen-bridge-audit/REPORT.md).

This verdict supersedes earlier conversational claims of full technical readiness. The audit sought failures rather than defending the existing model. The complete raw organizer specification was retrieved at current commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`; local copies and the original user brief were preserved. Source hashes are recorded.

## Critical issues and fixes

P1: feature interpolation/rolling/derivative state crossed period boundaries; stacking also lacked period-local windows. Independent synthetic mutation exposed the feature leak, and the repaired full predictor now passes exact other-half invariance. P1: ATD cache reused stale bytes when contents changed without increasing mtime; it now hashes XML and raw contents. P2: tied nearest-player speed depended on slot ordering; tied finite observations are averaged. P2: short periods crashed feature shape; rolling kernels now respect input length. The generator also rejects incorrect tactical frame counts, and metadata rejects unknown team references.

Pre-fix failures and post-fix execution receipts are retained. No P0 competition-rule violation was confirmed in the bounded audit. Public source access is still a submission condition, not silently marked complete.

The subsequent continuous loop also fixed P2 partial-coordinate visibility, lone-ball interpolation, nonfinite parser values and recovery from corrupted cache archives. Full-predictor synthetic contracts and all ten familiar external games show no regression. These repairs are not claimed as a significant quality gain. See CONTINUOUS_IMPROVEMENT.md for hypotheses, failing tests and decisions.

## Acceptance gates

| Gate | Result | Evidence / scope |
|---|---|---|
| 1 Rules compliance | CONDITIONAL | No prohibited target use found; external-only registries and published-anchor provenance checked. Original PFF access/terms and organizer access are qualified; team eligibility not independently verified. |
| 2 Data alignment | PASS for used inputs | ATD mapping, exact frame count, metre geometry, period isolation and resampling tests; no broadcast/homography dependency. |
| 3 Metric parity | PASS | Current official metric hash equals pinned; named synthetic edge cases and 100 random cases match legacy helper to 14 decimals. Actual evaluation uses official code. |
| 4 Temporal correctness | PASS for tested contracts | Full-predictor period isolation, 60 brute-force decoder cases, restart/transition contracts; one-second short-possession suppression explicitly retained as a quality limitation. |
| 5 Submission format | PASS | Exact audited output passes official CLI; strict two-column contract and mutation tests also pass. |
| 6 Reproducibility | PASS in tested environments; access conditional | Six identical outputs: two each on macOS ARM, Linux ARM and emulated Linux x86-64; fresh source and pinned dependencies, unchanged inputs. One host/operator; repository still private. |
| 7 Evidence quality | PASS with disclosed limits | Development, old-source final, synthetic and public diagnostic evidence distinguished; no claimed hidden score, guaranteed generalization or finalist probability. |

114 unittest cases pass on macOS and Linux ARM, zero errors/failures/skips, plus the existing 60 exhaustive-decoder comparisons. `test_receipt.json` and `test_receipt_linux.json` record test purpose, fixture/assertion location, actual status and rerun command. Initial corruption and Linux optional-dependency failures remain recorded. The test count is not a quality guarantee. Some requirements are conditional or not verified despite green tests; see REQUIREMENTS_COMPLIANCE.md.

## Measured quality

| Scope | Reference source | Spell F1 | Macro F1 | Interpretation |
|---|---|---:|---:|---|
| Corrected code, ten existing PFF development games, natural | Original PFF event annotations reconstructed via SDK | 0.655317 | 0.885608 | Familiar-data regression; mean spell change +0.000086 vs pre-audit |
| Same ten, fixed blind observation stress | Same independent reference, degraded geometry | 0.642574 | 0.877913 | Familiar-data regression; mean spell change +0.000088 |
| One frozen output, public five-minute window | Organizer's public corrected-tracking reference, 17 possessions | 0.600000 | 0.7548 | **DIAGNOSTIC ONLY — NOT A GENERALIZATION ESTIMATE** |
| Original source, previously sealed twelve-game PFF final, natural | PFF event-derived reference proxies | 0.604277 | 0.880511 | Original one-shot evidence; v4 spell 0.541801, 12/12 gains. Not rerun or retuned in this audit. |
| Corrected frozen source, six newly opened PFF bridges, natural | PFF event-state proxies, one familiar country per game | 0.613896 | 0.882073 | Prospective match holdout; v4 0.546054, 6/6 gains; no post-test tuning. |
| Same six, fixed blind observation stress | Same references, label-blind degraded geometry | 0.594795 | 0.870640 | V4 0.527951, 6/6 gains. |
| Corrected frozen source, all 36 PFF training matches | Same canonical reference policy | 0.625242 | 0.876044 | In-sample performance only; gap to new six +0.011345 Spell F1. |

All ten external games and both observation cases were retained; canonical and SDK-shadow results agree. No game loses Spell F1 in this correctness-fix regression. Tiny macro differences (~0.00002 natural / ~0.00004 blind mean loss) are retained. No improvement claim is made from these small changes. The previous twelve final matches were not reopened for corrected-model selection. Exhaustive cross-provider near-duplicate detection is not established. National-team overlap and only two final groups limit uncertainty estimates.

The published official baseline's full-match score 0.2433 is a reference from the organizers, not a local target test. External PFF scores do not establish superiority over it on the hidden competition match.

## Exact reproduction

Audited inference source commit: `1160825` (full SHA in submission_manifest.json). Later Linux-tested commits have identical inference-file hashes. Model weights remain boundary102; preprocessing/stacking/observation correctness repairs are explicit. Latest macOS runs take 33.280 / 25.567 seconds, Linux ARM 28.506 / 21.936, emulated Linux x86-64 59.649 / 42.986. All six official validators pass. The output is byte-identical to the preserved pre-audit CSV.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_mixed_candidate.py \
  --bundle assets/pff-mixed-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'Ruses Cules.csv'
```

CSV SHA-256: `cfbd16ea35075a199d2401c18f162d4c1f953be4049cd776d502ff42ea389c97`.

Package: `out/audit-ready/Ruses Cules.csv`, `out/audit-ready/Ruses Cules.md`, `out/audit-ready/submission_manifest.json`. Organizer source/data and per-frame predictions remain local and ignored. No public visibility change or organizer upload was performed.

## Submission checklist

- CSV: complete and official-format PASS; 199319 rows, exact expected bytes.
- Code and weights: inspectable on the private audit branch; public URL requirement remains pending user hold.
- Technical document: updated Markdown describes actual repairs, data provenance and limitations. Earlier PDF describes the original source and should not substitute for this audited document.
- Reproduction: locally demonstrated with separately installed macOS/Linux dependencies and fresh committed source; an independent hardware/operator rerun of corrected code remains unverified.
- PFF source: original-provider request path documented; licensing/access qualifications remain explicit, not inferred from a mirror.
- Team: three to five humans required; registered names/eligibility not independently verified here.
- Form: not submitted.

## Confidence assessment

Format correctness: high for the exact hashed file. Implementation correctness: strengthened by failing-then-passing synthetic tests, including partial XY/Inf/corrupted-cache contracts, but not exhaustive football semantic or numeric-corruption verification. Rule compliance: no prohibited-use evidence found within the inspected scope; source access and eligibility remain conditions. Reproducibility: directly demonstrated on macOS ARM, Linux ARM and emulated Linux x86-64 on one host. Generalization: historical external-match evidence plus six prospectively frozen new fixtures for the corrected source, with provider/proxy/shared-team limitations. No calibrated success probability is available.

The most likely quality failures remain short possessions, long missing-ball stretches, coordinate jitter, ambiguous control and provider-definition shift. A sparse four-player synthetic end-to-end fixture produced all D: its passing CSV validator proves execution only. Synthetic latent-state decoder tests must not be described as learned recognition of football semantics.

The added 22-player moving-controller fixture passes obvious uninterrupted H/A possession, but a ten-second ball dropout creates false D. This is preserved as a failure, and prompted the registered controlled-gap study. No claim of real-match improvement comes from the synthetic score. The fresh-data recheck found no unused SkillCorner games; SoccerTrack's event-anchored ball is ineligible as independent measured-ball evidence against targets from the same events.

Both controlled-gap candidates were rejected after all ten development games: natural Spell changes −0.000057 / −0.000385; blind changes zero. They are not in the submission. Continuous execution ends at condition B: confirmed implementation fixes and reproducibility work are complete, while a substantially different model cannot be safely promoted without further independent data/validated reference semantics. This is a concrete validation limitation, not a proof that quality cannot improve. The user's public-release/submission hold is preserved. See the stopping boundary in CONTINUOUS_IMPROVEMENT.md.

## Evidence index

- REQUIREMENTS_COMPLIANCE.md — requirement map and unverified conditions.
- DATA_ALIGNMENT_AUDIT.md — contracts, indices, dimensions, NaN and period handling.
- METRIC_PARITY_REPORT.md — exact metric and CSV differences.
- TEMPORAL_SEGMENTATION_AUDIT.md — output morphology and semantic limitations.
- FAILURE_ANALYSIS.md — severity, pre-fix evidence, repairs, remaining risks.
- REPRODUCIBILITY_REPORT.md — clean environment, repeated output and scan scope.
- TECHNICAL_DOCUMENT.md — organizer-facing implemented method and reproduction.
- CONTINUOUS_IMPROVEMENT.md — additional corrective iterations and model-quality hypothesis.
- FRESH_DATA_RECHECK.md — primary-source availability and circular-reference checks.
