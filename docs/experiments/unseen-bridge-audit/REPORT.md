# Six previously unopened matches: frozen-model validation

16 November 2026. **Completed real-match evaluation, not a unit-test proxy.** The already selected boundary102 model improves over the audited v4 control on all six newly opened fixtures. Mean official Spell F1 is **0.613896 vs 0.546054**, a gain of **0.067843**. The same current model scores **0.625242** on all 36 PFF training matches: the descriptive training-minus-new-match gap is **0.011345**.

This supports transfer of the existing improvement to unseen matches within PFF. It does **not** establish absence of overfitting, new-provider robustness, a hidden competition score, or a finalist probability. No model was trained, tuned, switched or promoted using this audit. The submission bytes are unchanged.

## Why these are new matches, and what independence means here

The original PFF registry partitioned 64 fixtures into 36 training, ten development, twelve original final and six country bridges excluded from **training and selection**. The earlier search stopped too early by treating that exclusion as a ban on evaluation. This audit corrects that decision without changing the old partition, reopening the twelve original final games, or allowing bridges into training/development tools.

All six bridges were registered before their contents were downloaded. The freeze was committed at source `237c04df092c933bd975c07db8689a4f919b6f56`, timestamp `2026-11-16T12:30:21.291983+00:00`. `freeze.json` hashes inference code, weights, readers, official metrics, reference policy and the plan. Existing raw/converted/reference content was checked absent. File IDs and fixture metadata were already catalogued; match contents had not been used. Predictions were saved before the reference adapter ran. Source hashes and prediction receipts are in `results.json`.

Each game has one country seen in PFF training/development and one country absent from those groups. The fixtures themselves do not overlap any training or development match. This is **match holdout with partial country overlap and the same provider**, not a fully unseen-team or unseen-provider benchmark. Shared teams link the six games into only four components. Provider event-state H/A/D references are independent of model predictions but remain proxies for football control, not manually corrected per-frame competition truth.

## All outcomes, with equal weight per whole match

The pinned official scorer computes Spell F1 at IoU thresholds 0.3/0.4/0.5/0.6/0.7. All recorded periods, including extra time, are included. All six references pass the predefined coverage checks: H/A/D present, periods 1/2 present, known reference coverage 100%. Canonical and SDK-shadow references produce identical results. There are 902,679 graded frames and 1,601 reference possession spells. No match was discarded.

| Fixture | Old v4 | Current boundary102 | Gain |
|---|---:|---:|---:|
| Morocco–Portugal (10512) | 0.4000 | 0.5493 | +0.1493 |
| Netherlands–Argentina (10511) | 0.5640 | 0.6164 | +0.0524 |
| Portugal–Switzerland (10509) | 0.6042 | 0.6384 | +0.0342 |
| Brazil–South Korea (10507) | 0.6747 | 0.6998 | +0.0251 |
| England–Senegal (10505) | 0.6236 | 0.6751 | +0.0516 |
| Netherlands–United States (10502) | 0.4100 | 0.5044 | +0.0945 |
| **Mean** | **0.546054** | **0.613896** | **+0.067843** |

| Scenario, canonical reference | Old Spell F1 | Current Spell F1 | Gain / positive games | Old macro F1 | Current macro F1 |
|---|---:|---:|---|---:|---:|
| Natural visible tracking | 0.546054 | 0.613896 | +0.067843 / 6 of 6 | 0.848335 | 0.882073 |
| Prespecified label-blind tracking degradation, seed 7 | 0.527951 | 0.594795 | +0.066845 / 6 of 6 | 0.839222 | 0.870640 |

The smallest natural gain is +0.025118, and the smallest degraded gain +0.032314. Current natural per-class frame F1 is H 0.855844, A 0.877893, D 0.912482. These frame metrics supplement, rather than replace, possession segmentation quality.

The descriptive 95% paired-match bootstrap interval for the natural gain is [+0.038483, +0.104674]. Resampling the four shared-team components gives [+0.038337, +0.085993]. Both use 10,000 draws and seed 20261116. Six games/four components and a common provider are too few for a calibrated universal generalization claim; these intervals describe this sample and its dependence assumptions.

## Training-versus-new-match check

The same frozen current inference was run on **all 36** original PFF training games, with no refitting. Geometry and reference caches match their original preflight hashes. Development results come from the already audited current-source regression, without another selection round. All figures below use natural tracking, canonical references and equal match weights.

| Group | Games | Spell F1 mean | Spell F1 range | Macro F1 mean | Meaning |
|---|---:|---:|---|---:|---|
| PFF training | 36 | 0.625242 | 0.442407–0.716832 | 0.876044 | In-sample, never a holdout |
| Reused PFF development | 10 | 0.655317 | 0.597938–0.701031 | 0.885608 | Familiar model-selection data |
| Newly opened bridge audit | 6 | 0.613896 | 0.504449–0.699772 | 0.882073 | Prospective match holdout |

Training-minus-new Spell F1 is +0.011345; development-minus-new is +0.041420. Macro F1 on the six new games is slightly higher than the training mean. These results do not show a large training-to-new-match collapse. They also show why the development mean should not be quoted as expected unseen performance. Different fixture difficulty, country mix, match stage and observation coverage contribute to group differences; the small train/test gap cannot isolate or exclude overfitting. This is an inference audit of the final ensemble, not an estimate of each component's fitting bias or a retrained cross-validation study.

`training_gap.json` retains all 36 per-game scores and their hashes. The old twelve-game final result (0.604277 on the original source) remains separate historical evidence, not pooled with this audit and not rerun.

## Failures remain visible

The following diagnostics pool counts across these six games and use natural tracking. Short-spell recall uses IoU 0.5 only; it is **not** the official multi-threshold Spell F1 and has no precision term.

| Reference possession duration | Reference spells | Matched by v4 | Matched by current | Current recall |
|---|---:|---:|---:|---:|
| Under 1 second | 54 | 0 | 0 | 0.0% |
| 1–3 seconds | 316 | 31 | 43 | 13.6% |
| 3–10 seconds | 566 | 245 | 281 | 49.6% |
| 10 seconds or longer | 665 | 499 | 548 | 82.4% |

The model still misses many short possessions; minimum-duration decoding and ambiguous/brief control are plausible contributors, not established sole causes. The under-one-second result is retained as a failure.

Visible-ball frame accuracy rises from 88.84% to 93.16%; missing-ball accuracy rises from 83.79% to 86.10%. Among the **196,486 reference-live frames with missing ball**, erroneous D falls from **43,553 (22.17%)** to **38,818 (19.76%)**. This is an important remaining error, despite the improvement. These are conditional frame diagnostics, not Spell F1 calculated on chopped missing-ball fragments. Visibility itself is not a random sample of game states.

## Decision and next boundary

**Retain the already selected model unchanged.** The hypothesis that its gain disappears on additional unused fixtures is not supported by these six outcomes. This was a validation-only study: the gain is against the old model, not a new improvement implemented during this audit. No post-test tuning is justified by these results.

The six games are now consumed evaluation data. They must not become a fresh final for a subsequently tuned candidate. The short-control and missing-ball weaknesses remain research priorities, but the prior boundary-event, duration-weighted and controlled-gap candidates already failed their development promotion criteria. Another safely promotable change requires a justified hypothesis and separate permitted evaluation evidence, not threshold searching against these six outcomes.

Technical format/reproduction evidence remains valid for the unchanged artifact. Overall submission conditions and the user's submission/publication hold remain separate; this report does not authorize release or claim unconditional readiness. Generalization confidence is stronger than before this audit, bounded by provider/reference dependence and six fixtures. A compatible new-provider full-match benchmark is still not established; see `SEARCH.md`.

## Reproduction and evidence

1. `plan.json`, commit 237c04d, and `freeze.json`, commit 7544033: prospective identities and exact frozen sources.
2. `tools/pff_unseen_audit.py run`: downloads the approved original sources, saves predictions before references, scores every registered game and checks the freeze. It deliberately refuses a completed output directory; `--resume` is for identical-source transport/completion recovery only.
3. `tools/report_unseen_audit.py`: verifies frozen identities and builds the paired/cluster analysis and source-hashed diagnostics from saved predictions.
4. `training_gap_plan.json` and `tools/audit_training_gap.py`: all 36 training fixtures, two workers, frozen current inference, no fitting. `tools/report_training_gap.py` verifies original reference receipts and compares the three separate groups.
5. `results.json`, `summary.json`, `training_gap.json`: published aggregates only. Raw tracking, references and per-frame predictions remain ignored local files. Reproduction needs the original-provider data access; a fresh clone alone is not sufficient.
6. New audit/report guard tests and the unchanged official submission gate are recorded in `verification.json`. The earlier complete inference test suite and six clean-environment reproduction runs remain documented in `validation/FINAL_AUDIT_REPORT.md`.
