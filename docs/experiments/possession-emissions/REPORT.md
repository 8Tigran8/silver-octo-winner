# Duration-weighted ownership emissions: completed development study

16 November 2026. **Neither registered replacement reaches the +0.020 meaningful-improvement gate. Retain boundary102.** The higher score is development evidence on familiar matches, not an independent final result.

## Method and isolation

The plan was committed as `72a2994` before code fitting and scores; the fitted source is commit `8489655`. A three-class geometry classifier directly estimates H/A/D instead of adjusting transition penalties. It uses the same 172 geometry/period-clamped temporal inputs as the boundary study, 200 fixed boosting iterations, one predeclared seed and no early stopping. Only the original 36 PFF training games are used. Known reference runs receive weight clip(50 / length-at-5Hz, 1, 10); unknowns receive zero. Fixed stride-five sampling at 5 Hz, equal game/scenario/swap mass and correctly swapped H/A targets yield 901,836 rows. No events, IDs, references or model outputs enter predictor features. References supply supervised targets and training weights only.

Two fixed policies mix 25% or 50% new joint H/A/D probabilities with frozen boundary102 probabilities; the original cost-9 decoder and one-second minima remain. This is a frame-emission experiment, not the proposed interval-level model. Its source/model/plan hashes and input lineage are preserved in the accompanying receipts. Training took 188.645 seconds locally.

## All results

Ten development games x two observation cases x two fixed references x three policies give 120 records. The unmodified control exactly reproduces previous spell, macro and per-class scores. Canonical and fixed SDK-shadow results agree. All games and both cases were retained.

| Policy | Natural Spell F1 | Gain | Wins | Blind Spell F1 | Gain | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Frozen boundary102 | 0.655231 | 0 | — | 0.642485 | 0 | — |
| emission25 | 0.663596 | +0.008365 | 9/10 | 0.651597 | +0.009112 | 9/10 |
| emission50 | 0.666112 | +0.010881 | 8/10 | 0.646699 | +0.004214 | 7/10 |

Both policies pass the 8-win natural requirement and the mean class/macro/blind regression limits, but fail the predeclared natural gain >=0.020. Emission25 gains natural macro F1 +0.003775 and blind +0.006107; emission50 gains +0.004984 and +0.006707. Every mean H/A/D F1 change is positive. Emission50 is stronger on natural observations; emission25 is stronger on blind corruption. No best scenario, game or post-hoc mixture is selected. No 75%/100% mixture search follows these results.

## Short-possession evidence

A separate descriptive audit gave the **old** probabilities the true start/end of each reference interval solely to understand errors. Among 416 missed 1–3-second possessions at IoU 0.5, only 29.8% select the true team using the mean probability over the correct interval (30.5% using summed log probabilities). This is oracle-conditioned analysis, not deployable performance or proof of an information-theoretic limit. It suggests that boundary timing alone cannot recover most of these misses with unchanged frame emissions. Full aggregates are in `evidence_audit.json`; no per-frame references are published.

## Decision

The registered gain gate fails. Do not run a reserve or promote a replacement. All 67 original frozen source/model/evidence files and the released CSV hash remain unchanged. The original twelve final matches, six excluded bridges and organizer data were not used in this study. Further training on these development outcomes cannot turn them into an independent final.

Three meaningful tests pass: duration weights reset at periods/exclude unknowns, team swapping is an involution with dead/unknown preservation, and joint fusion respects team symmetry and analytically known probabilities. The exact release CSV also passes the official validator again; that establishes format, not target label quality.

The frozen candidate remains technically suitable for a reviewed submission based on its original validation and reproduction evidence. It retains documented short-possession, coordinate-jitter and small-source overfit weaknesses. No hidden-match result, competitor benchmark or calibrated finalist probability is available. Publication and organizer upload remain on the existing user hold; team membership eligibility is separate.

Reproduction (requires the already documented authorized PFF caches and frozen boundary-study probability caches):

```sh
.venv/bin/python tools/possession_emission_study.py
.venv/bin/python tools/audit_short_possession_evidence.py
```

The study refuses to overwrite completed weights/scores. Weights and prediction arrays remain local. See `results.json` for every score and guard, `fit.json` for training input hashes, `model_receipt.json` for fitted model identity, and `READINESS.md` for criteria recorded before evaluation completed.
