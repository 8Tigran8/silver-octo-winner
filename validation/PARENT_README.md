# Independent audit status — 16 November 2026

This branch contains correctness repairs to period isolation, tracking caches, incomplete observations and equidistant slot handling. **CONDITIONALLY READY**: 114 tests pass on macOS and Linux ARM; six fresh-source runs across macOS ARM, Linux ARM and emulated Linux x86-64 reproduce the same CSV. Original final metrics belong to the pre-audit source; corrected development regression and source limitations are reported separately. See [final audit](validation/FINAL_AUDIT_REPORT.md), [continuous improvements](validation/CONTINUOUS_IMPROVEMENT.md), [audited technical document](validation/TECHNICAL_DOCUMENT.md), and [submission manifest](submission_manifest.json). Public repository access, source-access qualifications and team eligibility remain unresolved conditions. No organizer submission has occurred.

# Ruses Cules · More Than a Hack 2027

**Additional real-match audit:** the corrected frozen model was evaluated on all six previously unopened PFF bridge fixtures, without fitting or selection. Spell F1 is **0.613896 vs v4 0.546054**, with gains on 6/6 games; all 36 PFF training games score **0.625242** with the same current model. This strengthens within-provider unseen-match evidence, not a universal no-overfitting claim. Each new fixture shares one country with training/development, and references remain event-derived proxies. Full outcomes, training/test comparison and failures: [six-match audit](docs/experiments/unseen-bridge-audit/REPORT.md).

Ball possession from automatic tracking. **Current selected model: boundary102.** The organizer submission remains on hold. This branch makes the current code, frozen weights and evidence available for team review; it does not change repository visibility or submit the solution.

## Generate the current CSV

Use branch `release/independent-audit`; default `main` still contains historical v4 and does not represent this package. Use Python 3.12 and the organizer data bundle. The data directory must contain `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_mixed_candidate.py \
  --bundle assets/pff-mixed-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
```

The generator uses CPU inference with bundled weights. It needs no external match download, training cache, video, tagging or target ground truth. It refuses to overwrite an existing output.

Expected CSV SHA-256:

```text
cfbd16ea35075a199d2401c18f162d4c1f953be4049cd776d502ff42ea389c97
```

For two independent generations from committed source:

```bash
.venv/bin/python tools/reproduce_mixed_candidate.py --data-dir /absolute/path/to/data
```

The verifier requires a clean checkout and refuses to overwrite completed receipts. Two verified Apple M4 CPU runs produced identical bytes in 26.779 s and 26.838 s, with about 1.50 GiB peak child RSS. This verifies frozen-weight inference, not repeat training. VV also reports two byte-identical boundary102 CPU runs on Linux ARM (aarch64, Python 3.12.3, Torch 2.8.0), with both official format checks passing. He used an archive of review commit `072feab`, initialized locally as Git; this is teammate-reported archive-source reproduction, not an audited fresh upstream clone or Linux/x86 evidence. See [the scoped report](docs/experiments/pff-mixed/linux_arm_report.json).

## Method and measured quality

A 102-second temporal convolutional network, trained on 65 external games, is mixed 50/50 with the inherited boosted geometry path. A duration decoder produces H/A/D/X labels. The [technical document](docs/Ruses%20Cules.md) explains the model, all data sources, possession rules and reproduction.

| Same-game comparison | V4 spellF1 | boundary102 | Gain | Game wins |
|---|---:|---:|---:|---:|
| 29 reused development games, historical control | 0.61435 | 0.64317 | +0.02882 | 27/29 |
| 29 games with identical complete-fold exclusions | 0.61218 | 0.64317 | +0.03099 | 29/29 |
| 10 PFF development games | 0.61404 | 0.65523 | +0.04119 | 10/10 |
| 12 originally untouched PFF final games | 0.54180 | 0.60428 | +0.06248 | 12/12 |
| Same 12 final games, fixed label-blind sensor stress | 0.52648 | 0.58696 | +0.06048 | 12/12 |

All original predeclared development/final gates pass. Final teams from groups A/H are absent from PFF training/development. One candidate and all relevant source/model hashes were sealed before the final was opened. The reserve is now consumed; a replacement model needs a new untouched reserve. Scores use provider-event-derived reference proxies. Two shared-team groups limit statistical independence. Competition hidden-match quality and rank remain unknown.

Evidence: [65-game study](docs/experiments/pff-mixed/REPORT.md), [final results](docs/experiments/pff-mixed/final_results.json), [source/weight freeze](docs/experiments/pff-mixed/final_freeze.json), [exact-file receipt](docs/experiments/pff-mixed/reproduction.json), [original-requirement review](docs/experiments/pff-mixed/REQUIREMENTS.md).

## Material limitations

The [supplemental sensor pilot](docs/experiments/provider-stress/REPORT.md) found losses to same-exclusion v4 in all six distinct combined-noise cases on two familiar development games: 0.0064–0.0694 spellF1. One-factor probes retain advantages for cadence-only and burst-only changes but show sensitivity to independent coordinate jitter. These copies are not additional independent matches. All losses and duplicate SkillCorner presets are retained.

The standalone neural branch also shows a material Metrica full-fit/whole-fold-out gap: 0.8565 versus 0.7056 on only two games. Short possessions remain under-segmented. Neither the final gain nor the supplemental diagnostics establish absence of overfit or universal sensor robustness. No model settings were changed after final opening.

## Training data and access

The temporal branch uses 7 IDSSE/DFL games, 2 Metrica games, 20 SkillCorner A-League games and 36 PFF World Cup 2022 games. The inherited boosted and team components retain their 9-game and 19-game registries, which overlap the 65 games. No labels from the provided match enter training, validation, selection or calibration.

**Original PFF provider access was obtained on 15 November 2026.** It is available through the provider's request form. The separate Hugging Face mirror remains unused. [PFF access and source terms](docs/PFF_SOURCE.md) explains the original route, event-derived reference construction and the distinction between original provider access and the mirror's GPL-3.0 label. Raw PFF files and vendor specification PDFs are not redistributed.

Training is optional for generating the CSV. Research reproduction additionally needs public source files and the audited training caches. Start with [the registered training plan](docs/experiments/pff-mixed/plan.json), [original split](docs/experiments/more-data/pff_protocol.json) and source manifests. Entry points are `tools/fetch_pff_training.py`, `tools/pff_mixed_train.py prepare`, and `tools/pff_mixed_train.py fit --full --device mps`. The full audit uses complete-fold exclusions in every direct and nested learned component and train-only normalization. Retraining on a different backend is not promised to reproduce weight bytes.

## Submission contract and data handling

The [official challenge](https://github.com/metrica-sports/more-than-a-hack-2027) requires a UTF-8 CSV with header `frame,label`, exactly 199,319 ordered frames (0 through 199318), and labels H/A/D/X. SpellF1 determines ranking. The exact CSV, public source repository and short technical document must describe the same algorithm. The organizer's welcome email sets the deadline at **17 November 2026, 23:59 CET**.

Organizer tracking, video, tagging, homography and target labels stay local. Do not commit or redistribute them, derived tracking exports, credentials, raw external matches or reference arrays. Do not upload organizer materials or extracts to hosted LLMs. Source, permitted frozen models and aggregate experiment receipts are the shareable package. Repository publication and the organizer form remain separate pending actions.

## Historical versions

`run.py v4` still generates the historical control, not the current candidate. Its [method](docs/experiments/temporal/TECHNICAL_V4.md) and assets remain available. Earlier v1/v2/v3 releases, failed studies, training recipes and receipts are preserved in [HISTORY.md](HISTORY.md) and their experiment directories.

## Later research

The separate boundary-event head on `release/boundary-events` failed its predeclared meaningful-improvement gate (+0.00297 Spell F1, 7/10 development wins; required +0.020 and 8/10). It is not part of this release. See [the complete negative result](https://github.com/8Tigran8/silver-octo-winner/blob/release/boundary-events/docs/experiments/boundary-events/REPORT.md).
