# Current model on all 18 previously opened matches

16 November 2026. **Completed fixed-model re-evaluation, not a fresh independent test.** The user explicitly requested reopening all existing test fixtures after the current model was adopted. The 18 matches stayed out of weight fitting and numerical candidate selection. Their earlier parent outcomes and weaknesses were already known and motivated the research direction; this is disclosed rather than called a pristine holdout. No organizer labels were read or scored.

## Protocol and result

One frozen current model (weighted_short_gap) was compared with boundary102, both using the same current audited preprocessing, original raw inputs, period boundaries and official scorer. All 18 matches, both natural/fixed blind scenarios and both canonical/SDK-shadow references were retained. No settings, weights, decoding thresholds, references or output labels were changed after outcomes. Plans, source/model/raw hashes and historical output hashes were frozen before prediction.

| Cohort | Scenario | Parent Spell F1 | Current Spell F1 | Gain | Wins | Worst gain | Review guards |
|---|---|---:|---:|---:|---:|---:|---|
| bridge6 | natural_detected | 0.613896 | 0.638797 | +0.024901 | 6/6 | +0.010848 | PASS |
| bridge6 | label_blind_seed7 | 0.594795 | 0.616778 | +0.021982 | 6/6 | +0.003483 | PASS |
| original_final12 | natural_detected | 0.604159 | 0.639452 | +0.035293 | 12/12 | +0.014969 | PASS |
| original_final12 | label_blind_seed7 | 0.586834 | 0.621332 | +0.034499 | 12/12 | +0.020639 | PASS |
| all 18 | natural_detected | 0.607404 | 0.639234 | +0.031829 | 18/18 | +0.010848 | PASS |
| all 18 | label_blind_seed7 | 0.589488 | 0.619814 | +0.030326 | 18/18 | +0.003483 | PASS |

Review guards were declared before this run: each cohort/scenario/reference must have nonnegative mean Spell change, mean macro change≥−0.002, and worst game Spell change≥−0.020. These are regression review rules, not a statistical proof of generalization.

## Every match

| Cohort / fixture (PFF ID) | Parent natural | Current natural | Natural gain | Parent blind | Current blind |
|---|---:|---:|---:|---:|---:|
| bridge6: Netherlands–United States (10502) | 0.504449 | 0.548485 | +0.044036 | 0.469461 | 0.516828 |
| bridge6: England–Senegal (10505) | 0.675113 | 0.685961 | +0.010848 | 0.642759 | 0.658009 |
| bridge6: Brazil–South Korea (10507) | 0.699772 | 0.721088 | +0.021316 | 0.666667 | 0.682028 |
| bridge6: Portugal–Switzerland (10509) | 0.638356 | 0.661024 | +0.022668 | 0.636697 | 0.652679 |
| bridge6: Netherlands–Argentina (10511) | 0.616421 | 0.633884 | +0.017463 | 0.616807 | 0.620290 |
| bridge6: Morocco–Portugal (10512) | 0.549266 | 0.582341 | +0.033075 | 0.536383 | 0.570833 |
| original_final12: Senegal–Netherlands (3812) | 0.640000 | 0.666408 | +0.026408 | 0.622044 | 0.647801 |
| original_final12: Qatar–Ecuador (3814) | 0.662385 | 0.684581 | +0.022196 | 0.643519 | 0.665789 |
| original_final12: Uruguay–South Korea (3825) | 0.573876 | 0.600411 | +0.026535 | 0.556223 | 0.591903 |
| original_final12: Portugal–Ghana (3826) | 0.678632 | 0.701856 | +0.023223 | 0.654077 | 0.683817 |
| original_final12: Qatar–Senegal (3829) | 0.642348 | 0.690619 | +0.048271 | 0.632985 | 0.658683 |
| original_final12: Netherlands–Ecuador (3830) | 0.593939 | 0.626357 | +0.032417 | 0.575200 | 0.623002 |
| original_final12: South Korea–Ghana (3841) | 0.596364 | 0.642581 | +0.046217 | 0.597691 | 0.637199 |
| original_final12: Portugal–Uruguay (3843) | 0.604702 | 0.628070 | +0.023369 | 0.596709 | 0.620641 |
| original_final12: Ecuador–Senegal (3844) | 0.534381 | 0.577281 | +0.042900 | 0.525737 | 0.564662 |
| original_final12: Netherlands–Qatar (3845) | 0.572381 | 0.614044 | +0.041663 | 0.502415 | 0.553271 |
| original_final12: Ghana–Uruguay (3856) | 0.606294 | 0.621262 | +0.014969 | 0.583746 | 0.604384 |
| original_final12: South Korea–Portugal (3857) | 0.544601 | 0.619956 | +0.075355 | 0.551659 | 0.604835 |

## Pooled error diagnostics across all 18 matches

| Scenario / model | Short spells<3 s matched / reference (IoU0.5) | False D / missing-ball live frames | False live / missing-ball D frames | Wrong team / missing-ball live frames |
|---|---:|---:|---:|---:|
| natural_detected / boundary102 | 123/1216 (10.12%) | 126855/579297 (21.90%) | 40504/1070662 (3.78%) | 62047/579297 (10.71%) |
| natural_detected / weighted_short_gap | 173/1216 (14.23%) | 109345/579297 (18.88%) | 27983/1070662 (2.61%) | 64393/579297 (11.12%) |
| label_blind_seed7 / boundary102 | 108/1216 (8.88%) | 156142/637180 (24.51%) | 38777/1074300 (3.61%) | 63426/637180 (9.95%) |
| label_blind_seed7 / weighted_short_gap | 160/1216 (13.16%) | 117002/637180 (18.36%) | 29259/1074300 (2.72%) | 70404/637180 (11.05%) |

Counts pool frames/spells, unlike the equal-match headline average. Short recall has no precision term. Missing-ball wrong-team errors increase even while false stoppages and total error decrease; both sides are disclosed.

## Current train / development / audit comparison

| Group | Games | Current natural Spell F1 | Interpretation |
|---|---:|---:|---|
| PFF training |36|0.652584|In-sample|
| PFF development |10|0.680009|Reused model-selection set;22 changed policies|
| bridge6 |6|0.638797|Previously opened matches, no fitting on them|
| original_final12 |12|0.639452|Previously opened matches, no fitting on them|
| all 18 |18|0.639234|Previously opened matches, no fitting on them|

The training score was recomputed with the exact current portable model on all 36 original PFF training games, not copied from its parent. The model itself was not retrained. The group gap combines fitting optimism, match difficulty, teams and measurement conditions; small or negative train-minus-test differences cannot prove absence of overfitting. The current 18-game comparison is positive or negative evidence for these matches only; repeated development selection and shared event-proxy semantics remain limitations.

## Why Netherlands–United States is difficult

Spell F1 improves 0.504449→0.548485; macro F1 0.849498→0.869916. The reference contains 309 possessions; the parent predicts 208, current 219. This is substantial under-segmentation, with missed brief switches and inaccurate extents, not evidence that the metric is capped near0.5.

| Reference possession length | Reference count | Parent matched at IoU0.5 | Current matched at IoU0.5 |
|---|---:|---:|---:|
| under_1 s | 10 | 0 | 0 |
| 1_to_3 s | 66 | 3 | 6 |
| 3_to_10 s | 115 | 39 | 47 |
| 10s_plus | 118 | 85 | 95 |

| Error on Netherlands–USA | Parent | Current |
|---|---:|---:|
| visible: false_dead / reference live frames | 251/62825 (0.40%) | 372/62825 (0.59%) |
| visible: wrong_team / reference live frames | 4207/62825 (6.70%) | 3091/62825 (4.92%) |
| visible: false_live / reference dead frames | 1822/5018 (36.31%) | 1334/5018 (26.58%) |
| missing: false_dead / reference live frames | 7590/32145 (23.61%) | 6670/32145 (20.75%) |
| missing: wrong_team / reference live frames | 4881/32145 (15.18%) | 5120/32145 (15.93%) |
| missing: false_live / reference dead frames | 1889/46798 (4.04%) | 1168/46798 (2.50%) |

Some conditional errors worsen: wrong-team frames during ball gaps increase, and visible-ball false-D frames increase. These losses are not hidden by the aggregate improvement.

At IoU0.5 the current model misses 161of309 reference possessions. Descriptive majority-label attribution separates 33 mostly called D,62 mostly assigned to the opponent, and 66 mostly assigned to the correct team but with unsuitable extent or one-to-one competition. These are measured error categories, not proven causal mechanisms. Of the missed possessions mostly called D, mean ball visibility is only 5.94%.

Ball is missing on 53.8% of graded frames, similar to England–Senegal53.9% and  Brazil–South Korea53.6%. Therefore missingness rate alone does not explain the lower score. Current wrong-team error on reference-live visible frames is 4.92%, versus 3.21% and  2.60% on those two matches. Several error sources coexist.

First-half Spell F1 slightly declines 0.587013→0.583607; second-half improves 0.437762→0.518310. The aggregate gain does not imply every period improves. At stricter IoU0.7 only 102of309 reference possessions match, versus 148atIoU0.5. Boundary extent remains a material weakness.

The source clock check reports maximum error below 0.000001 s, unchanged raw hashes and full reference coverage. The matched typed-transition median latency is 0 frames for the current model (only among matches within±1 s), so this does not suggest one simple global offset; that statistic cannot rule out large errors among unmatched transitions. No label-based time-shift tuning was performed. Automatic event-state references may differ from actual controlled possession; no video-based tactical explanation is invented.

## Leakage, implementation and reproducibility

The 36/10/12/6 PFF groups have empty match-ID intersections. Both current training registries exactly match the approved 65/36 sets. Current normalization is byte-for-byte equal to the parent train-only normalization. Geometry excludes provider possession/event fields and estimated ball positions; unit tests perturb annotation fields without affecting geometry. Augmentation uses dummy labels for audit degradation. Inference receives independent source period spans, not reference-derived score windows. Actual references are loaded after prediction bytes are saved. These checks are bounded evidence against direct leakage, not proof against every possible leakage path or exhaustive cross-provider duplicate detection.

For Netherlands–USA, paired evaluation asserts zero probability difference against both standalone current and parent entrypoints in both scenarios before saving outputs. The parent is recomputed on all 18 matches; differences from historical parent arrays/scores are explicitly retained in source receipts. The original 12 historical numbers are not silently substituted for current-source controls. Canonical and SDK-shadow reconstruction agreement is checked in complete scores.

A diagnostic-only implementation error occurred: the blind augmentation helper omits period metadata, and the new reporting function initially expected it. Predictions had already used the independent original periods correctly and were saved before the error. reporting_amendment.json preserves original/new tool hashes; original freeze, failure log and all saved prediction/reference bytes are retained. Reporting now explicitly supplies original periods and resumes from saved arrays. A dedicated synthetic regression verifies exact saved prediction preservation and correct period diagnostics. No model, input geometry, reference policy or candidate choice was changed.

The full current test suite and exact submission gate are rerun after this repair; see verification.json and tests.txt. Previous clean-source generation evidence remains valid because inference files, model bytes and the target CSV are unchanged. Four macOS/Linux outputs were byte-identical and official-validator valid. The target CSV SHA-256 remains 1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0.

## Reproduction and decision

Original source and raw files must be available locally; raw/reference/prediction arrays remain ignored and are not redistributed. The evidence directory publishes aggregate metrics and hashes only. tools/recovery_opened_audit.py run --resume completes the frozen 18-game audit using the reporting amendment; tools/recovery_training_audit.py run --resume completes the 36-game training inference audit; tools/report_recovery_opened.py recreates this report. Completed evidence is not replaced with a different candidate.

All per-game regressions, failed guards and technical defects must be investigated before claiming readiness. All frozen review guards pass: retain the already selected model. These reopened outcomes do not authorize a new threshold search. Remaining weaknesses are short-possession recall, ambiguous ownership, long observation gaps, domain shift and proxy labels. Repeatedly searching these18 matches would consume them as further development. A genuinely new provider-diverse labeled sample would still be needed for stronger prospective generalization evidence. No organizer submission or repository publication was performed.
