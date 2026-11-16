# Ruses Cules: boundary102 technical method

Updated 16 November 2026. Current source, frozen model and candidate CSV are prepared for team review. The organizer submission and public release remain pending. No competition-match quality score is claimed.

## Method and supplied assets

The algorithm assigns H/A/D/X to exactly 199,319 ordered frames using the supplied automatic tactical XML/TXT. Tracking IDs 1 through 199319 map to video frames 0 through 199318. Video, homography and Smart Tagging are unused. No human labels, public/private target ground truth or manual corrections of the provided game enter fitting, selection or calibration.

```text
Automatic tracking -> 87 geometry + 21 contact features
                   -> frozen 9-game boosted path + 19-game team head
                   -> 108 features + 108 missing masks -> 102-second TCN
                   -> fixed 50/50 probability mixture -> duration decoder
                   -> H/A/D/X CSV
```

Missing ball observations make nearest-player rules unstable. The temporal network uses surrounding observations and team context to retain possession through occlusions; an auxiliary change loss encourages useful boundaries. The inherited boosted path supplies a complementary geometry estimate. This creates a possession timeline for match analysis from existing tracking, without requiring a new video detector.

The TCN has 48 channels, dropout 0.15, and seven residual blocks with two 3-tap convolutions per block. Dilations 1/2/4/8/16/32/64 give 509 steps at 5 Hz: 101.8 seconds of receptive field. Two logits predict dead probability and home ownership conditional on live play. Direct and team-swapped predictions impose home/away symmetry. Both probability channels are mixed 50/50 with the inherited v3 path. The original v4 temporal branch is a comparison control.

The duration Viterbi decoder retains cost-9 transitions and one-second minimum live/dead intervals. Internal states are H, A, dead-after-H and dead-after-A; both dead states output D. Loose balls and challenges use learned contact/context probabilities and the transition penalty. A nearest touch does not automatically transfer possession. A stoppage separates possessions even when the same team restarts. Restart snapping is disabled. Published period/kickoff alignment is retained. Inference is offline and uses future context.

## Training data, annotation and access

The temporal branch trains on 65 external games, all outside the provided game. The inherited boosted and team components retain their separate 9-game and 19-game registries, which overlap this corpus.

| Source | Games | Unaugmented H/A/D frames | Reference and source terms |
|---|---:|---:|---|
| IDSSE / DFL | 7 | 1,002,644 | Provider BallStatus/BallPossession; CC BY 4.0 |
| Metrica samples 1/2 | 2 | 286,162 | Provider events converted by code to possession intervals; event-derived proxies; acknowledge Metrica |
| SkillCorner A-League | 20 | 2,955,525 | Automatic H/A phases; metadata-period complement is a D proxy; MIT; credit SkillCorner/PySport |
| Original PFF World Cup 2022 | 36 | 5,636,115 | Embedded provider events reconstructed using primary SDK semantics; original provider access by request; no raw redistribution |

Total: 9,880,446 unaugmented reference-period frames. Frames and augmented copies are not independent matches. Provider annotation workflows were not independently audited. Validation-only sources also include Metrica sample 3, one Groningen/SciSports WCSF sample and nine European SkillCorner games; the last have conditional H/A references only.

Sources: [IDSSE/DFL paper](https://doi.org/10.1038/s41597-025-04505-y), [Metrica sample data](https://github.com/metrica-sports/sample-data), [SkillCorner Open Data](https://github.com/SkillCorner/opendata), [PFF original provider release](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset).

**Original PFF access was obtained on 15 November 2026 through the provider's form.** Organizers can use the same request route. The original article states no standard dataset license; the GPL-3.0 label on the separate Hugging Face mirror is not treated as a license for the original release. That mirror remains unused. [The access record](data/PFF.md) explains the route and source terms. Raw files and vendor PDFs are not committed.

PFF predictors use only raw VISIBLE observations. ESTIMATED tracks, event-informed smoothed coordinates, event/possession fields and identities are excluded from features. References are reconstructed separately: OUT/END means dead, kickoff/OTB means live, and home_ball supplies ownership. Period state resets and duplicate-conflict masking are fixed. A fixed SDK last-row shadow checks sensitivity. These are provider-event-derived approximations, not corrected frame-by-frame competition truth.

## Fitting and isolation

Normalization uses training observations only, with a 0.01 standard-deviation floor, clipping to [-8, 8] and explicit missing masks. Training uses seed 20261113, 6,000 AdamW updates, learning rate 0.001, weight decay 0.001, batch 16, 256 supervised 5-Hz steps and 256 context steps on each side. Provider groups are sampled equally, then games/scenarios uniformly. Team swaps are random. The boundary BCE weight is 0.15, with positive weight 8; unknown/context-only adjacent pairs are masked. SkillCorner D-proxy loss has weight 0.25.

All direct and nested learned components and normalizers exclude the complete scored fold on the reused 29-game/67-cell grid. Actual normalizers and checkpoint lineage are verified. The original PFF split is 36 training, 10 development, 12 final and 6 excluded country bridges. Both national teams in final groups A/H are absent from PFF training/development. One candidate and source/model/reader/scenario hashes were sealed before opening final contents. No model switch or post-test tuning followed.

## Fixed evaluation and results

The unmodified official spellF1 metric matches maximal H/A possessions at temporal IoU thresholds 0.3/0.4/0.5/0.6/0.7 and averages F1. MacroF1 and H/A/D F1 guard frame quality. Compare models on the same games; absolute values across different cohorts are not train/test gaps.

| Comparison | V4 | boundary102 | Gain | Game wins |
|---|---:|---:|---:|---:|
| 29 reused development games, historical control | 0.61435 | 0.64317 | +0.02882 | 27/29 |
| 29 games, identical complete-fold exclusions | 0.61218 | 0.64317 | +0.03099 | 29/29 |
| 10 PFF development games, natural | 0.61404 | 0.65523 | +0.04119 | 10/10 |
| 12 originally untouched final games, natural | 0.54180 | 0.60428 | +0.06248 | 12/12 |
| Same 12 final games, fixed label-blind sensor stress | 0.52648 | 0.58696 | +0.06048 | 12/12 |

All original predeclared source/cohort, macro, class, stress, lower-tail, large-loss and final gates pass. Final natural macroF1 gain is +0.02572; H/A/D F1 gains are +0.02577/+0.03175/+0.01964. Group gains are A +0.07076 and H +0.05420. The descriptive paired whole-game 95% gain interval is [0.04827, 0.08266]. Shared teams and only two groups limit independence assumptions. Canonical and fixed SDK-shadow results agree.

The data-expanded 25-second control also passes development, but the longer-context/boundary variant wins the predeclared selection rule. Those two changes are not separately ablated. All eleven reused external guards pass. Losses remain on noisy Metrica3 spellF1 (-0.00900) and European conditional H/A macroF1 (-0.00365), within the original 0.01 guard. [The full study](experiments/pff-mixed/REPORT.md) preserves every candidate, prior failure, split and source identity.

## Overfit and sensor limitations

| Standalone neural branch | Full-fit spellF1 | Whole-fold-out | Gap |
|---|---:|---:|---:|
| DFL, 7 games | 0.68989 | 0.66764 | +0.02225 |
| Metrica, 2 games | 0.85653 | 0.70556 | +0.15097 |
| SkillCorner, 20 games | 0.70864 | 0.71486 | -0.00622 |

The Metrica gap is a material source-specific overfit risk. This is a descriptive post-selection diagnostic of the neural branch; deployment uses a 50/50 fusion. The independent final supports generalization beyond memorizing training games, but cannot prove absence of overfit everywhere.

A later [two-game sensor pilot](experiments/provider-stress/REPORT.md) loses to same-exclusion v4 in all six distinct combined perturbation cases: -0.00638 to -0.06944 spellF1. Clean/native controls improve. Post-hoc one-factor probes retain gains for cadence-only/burst-only changes but lose under independent coordinate jitter. These familiar-game copies are supplemental diagnostics, not new independent matches or a new promotion gate. No inference policy or candidate CSV changes follow.

Short possessions remain under-segmented: the PFF development predicted/reference spell-count ratio is 0.755, with mean predicted duration 17.54 s versus reference 13.75 s. Long gaps, incomplete players, ambiguous control, annotation differences and anonymous WCSF provenance remain limitations. Hidden competition quality and qualification probability are unknown.

Further work would improve jitter robustness using development-only augmentation, add independent Metrica-like games, reduce tiny-source sampling weight and improve short-spell duration modelling. Any replacement requires a new untouched reserve; the consumed final cannot become a tuning set.

## Exact-file reproduction

Use Python 3.12 and the pinned dependencies. No external training download/cache or retraining is needed to generate labels; frozen weights are bundled. Model, metadata and inherited hashes are checked before loading. The new checkpoint uses weights-only deserialization.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_mixed_candidate.py \
  --bundle assets/pff-mixed-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'Ruses Cules.csv'
```

Two clean committed-source Apple M4 CPU generations passed the official validator and produced identical bytes in 26.779 s and 26.838 s, with about 1.50 GiB peak child RSS. This repeats frozen-weight inference, not training. Input/model hashes remained unchanged. Tested inference source: `d10dd97438c3`; all 67 frozen files remained unchanged through final validation and later sensor diagnosis. The deployment loader also exactly reproduces previously scored PFF development predictions.

VV independently reported byte-identical Linux ARM reproduction of the historical v4 CSV. That is not yet a Linux reproduction result for boundary102. Current-platform verification remains explicit in the receipts.

CSV SHA-256: `cfbd16ea35075a199d2401c18f162d4c1f953be4049cd776d502ff42ea389c97`.

The four-page PDF and exact CSV are prepared locally under `out/review-ready/`. The organizer form has not been submitted. Historical v4 remains available under its explicit command and [archived method](experiments/temporal/TECHNICAL_V4.md).
