# Short-possession and missing-ball recovery: development experiments

16 November 2026. Historical first-stage result; the subsequent [weighted composition study](../spell-weighted-finetune/REPORT.md) qualifies and is now adopted. Statements about unchanged production below describe this first-stage checkpoint.

**Completed: 16 changed policies plus the unchanged control, all ten original development games, two observation scenarios and two reference reconstructions. No candidate reaches the unchanged +0.02 Spell F1 promotion gate in both scenarios. Production inference, weights and submission remain unchanged.**

The strongest minimum-scenario gain comes from `neural_expert25`: natural Spell F1 0.655317 → 0.666956 (+0.011639, 9/10 wins), blind 0.642574 → 0.654964 (+0.012391, 10/10 wins). This is positive reused-development evidence and a retained research prototype, not an approved production improvement or independent generalization result.

## Scope and controls

Only the original 36 PFF training matches and ten development matches are whitelisted by the experiment entrypoints. The six recently evaluated bridges, original twelve final matches and provided-match labels were not used for any candidate comparison, feature, training, selection or calibration in this study. The same corrected frozen boundary102 control reproduces its prior development Spell F1, macro F1 and per-class results exactly. Canonical and SDK-shadow references are retained.

The original plan and implementation were committed before candidate scoring. Two later hypotheses were registered separately after observing earlier development outcomes; this adaptive order is explicit in `complement_plan.json` and `contact_plan.json`. No original gate, numerical setting or completed result was overwritten. This is not a fresh holdout, and positive descriptive confidence intervals after this development search are not calibrated no-overfitting guarantees.

## Iterations and decisions

1. **Duration/transition hypothesis.** The one-second minimum may suppress real brief ownership. Test .4 seconds alone, then .4 seconds with H/A switch cost4.5, preserving D duration/costs. A synthetic brief-control case is recovered, but real-match gains are negligible or inconsistent. Reject both for production.
2. **Missing-ball evidence hypothesis.** The old boosted D head may overinterpret missing coordinates. Use the already trained neural D head in missing-ball frames only; separately test bounded live-endpoint log-odds correction without filling coordinates. Neural gating helps, but increases missed true stoppages and falls below the gate. Endpoint correction harms both scenarios. Reject for production.
3. **Player activity hypothesis.** Train a separate regularized binary D expert from 64 period-local player activity features on all36 original training games, natural and fixed blind data, missing-ball frames only. No ball coordinates, visibility duration, model outputs, identities or timestamps enter this head. There are676,547 sampled training rows, equal match weights,150 fixed HGB iterations. Test .25/.5 blends and fixed short-decoder combinations. Some gains occur, but the expert alone does not fix false D and fails the gate.
4. **Complementary errors.** Neural gating reduces false D but misses more actual stops; the activity head has the opposite effect. Register exactly .75 neural/.25 activity D in gaps, with original or the already fixed short decoder. The original-decoder combination is the strongest stable research candidate, but +0.011639/+0.012391 remain below +0.02.
5. **Visible contact.** Duration relaxation barely changes results, suggesting insufficient brief-control evidence before decoding. Register measured close-contact log-odds correction under continuous visible ball, relative velocity, opponent separation and player visibility checks. Actual-match tests decrease Spell F1 despite passing geometry/symmetry/period tests. Reject all contact additions.

No failed policy is called an improvement merely because it passes a unit test, recovers more short spells, or reduces one conditional error. Full Spell F1 and countervailing errors determine the decision.

## All registered policy outcomes

Gates, each scenario and reference: mean Spell gain≥0.02; at least8/10 positive games; worst game≥−0.02; mean macro gain≥−0.002. Every changed policy below fails at least the mean Spell gate.

| Policy | Natural Spell gain | Blind Spell gain | Natural wins | Blind wins | Worst natural game |
|---|---:|---:|---:|---:|---:|
| control | +0.000000 | +0.000000 | 0/10 | 0/10 | +0.000000 |
| short04 | -0.000142 | +0.000000 | 0/10 | 0/10 | -0.001421 |
| short04_ha45 | +0.001877 | -0.000142 | 5/10 | 6/10 | -0.007890 |
| neural | +0.009161 | +0.012093 | 9/10 | 9/10 | -0.000840 |
| bridge | -0.000308 | -0.000598 | 3/10 | 2/10 | -0.005894 |
| short_neural | +0.009873 | +0.012419 | 8/10 | 9/10 | -0.006534 |
| short_bridge | +0.001572 | -0.001022 | 5/10 | 5/10 | -0.011288 |
| expert25 | +0.005288 | +0.006608 | 7/10 | 10/10 | -0.003299 |
| expert50 | +0.003567 | +0.003525 | 6/10 | 7/10 | -0.010723 |
| short_expert25 | +0.007222 | +0.005337 | 7/10 | 8/10 | -0.008479 |
| short_expert50 | +0.004069 | +0.002634 | 5/10 | 6/10 | -0.008231 |
| neural_expert25 | +0.011639 | +0.012391 | 9/10 | 10/10 | -0.004979 |
| short_neural_expert25 | +0.011118 | +0.013022 | 7/10 | 8/10 | -0.016078 |
| contact | -0.004314 | -0.003445 | 1/10 | 3/10 | -0.013423 |
| short_contact | -0.006869 | -0.006210 | 2/10 | 3/10 | -0.028732 |
| contact_neural_expert25 | +0.007406 | +0.008817 | 7/10 | 9/10 | -0.019114 |
| short_contact_neural_expert25 | +0.001904 | +0.007055 | 7/10 | 6/10 | -0.039511 |

## Error tradeoffs of the best research prototype

Natural tracking: reference-live missing-ball false D decreases from71,319/339,698 (20.99%) to59,210/339,698 (17.43%). Missing-ball true-stoppage false-live rate changes from4.42% to4.41%. Macro F1 improves from0.885608 to0.892629.

Blind tracking: missing-ball false D decreases from22.14% to16.07%, but missed true stoppages increase from4.21% to4.76%. Macro F1 improves from0.877913 to0.889210. This deterioration is retained rather than hidden behind the overall gain.

Possessions shorter than3 seconds, natural tracking: current control matches65/578 at IoU0.5 (11.25%). Best gap prototype matches70/578 (12.11%). Adding the short decoder matches81/578 (14.01%), but only7/10 natural games improve and one loses0.016078. These recall-only diagnostics do not include precision and cannot replace official multi-threshold Spell F1.

For `neural_expert25`, descriptive paired-match bootstrap95% gain ranges are [+0.005340,+0.018112] natural and [+0.007486,+0.016821] blind. Shared provider/team dependencies and post-selection reuse of ten games limit interpretation.

## Reproduction and next experiment

`tools/short_gap_study.py prepare-dev`, `prepare-train`, `fit`, `mechanical`, `expert` implement the initial registered study. `tools/report_short_gap.py` checks a complete unique outcome grid and reports all gates. `tools/short_gap_complement.py` and `tools/contact_recovery_study.py` implement the separately frozen follow-ups. Frozen inputs, code and weights are checked before execution; completed outcomes are not overwritten. Raw/cache/prediction files stay local under ignored reports/. Public artifacts contain only code, plans, hashes and aggregate metrics.

`scores.json`, `complement_scores.json`, `contact_scores.json` together contain680 unique match/scenario/reference/policy records; `contact_result.json` summarizes all17 policies. Failed experiments are retained for reproducibility, not imported into production. The activity model and exact preparation receipts remain local; `expert_meta.json` records its training provenance.

This study does not end the improvement loop. A separately registered matched neural continuation examines the training objective: current H/A loss weights frames uniformly, so brief possessions contribute little. Compare equal additional training with unchanged loss against training-only short-spell/missing-observation weights, using freshly regenerated corrected features on the same65 approved training games. See `../spell-weighted-finetune/plan.json`; no result is assumed in advance.
