# Ruses Cules: estimating team possession from tactical tracking

## Approach

We estimate home possession (H), away possession (A) and dead ball (D) from the supplied tactical ATD XML/TXT. We use player and ball geometry, a temporal network and a player-activity model, then decode a sequence of possessions with duration constraints. This combines contact evidence when we can see the ball with player movement during tracking gaps. We considered a separate possession-boundary detector, but its small development gain did not justify adding it.

The selected model is `weighted_short_gap`. We froze its weights and decoding parameters before the final comparison. The CSV contains 199,319 rows, frames 0–199318, with the header `frame,label`. We map ATD frame IDs to zero-based output indices. X covers frames outside the playing blocks; published period boundaries and second-half kickoff determine that coverage.

We use the tactical ATD assets for inference. Video, homography, Smart Tagging, target possession labels and human-produced events about the supplied match do not enter this release's inference, training or model selection. Historical parent experiments have separate records in the repository; their public-window scores do not establish this release's quality.

## Model and possession rules

We convert coordinates to metres, validate track/team mappings and reset interpolation, derivatives and temporal context at each period. A ball observation requires two finite coordinates. We retain a missing-ball indicator through preprocessing.

The geometry network uses 108 features plus finite masks at 5 Hz: player–ball distances, nearby counts, team shape, speed, observation gaps and contact context. It has width 48, seven residual dilation blocks and 101.8 seconds of context. We average direct and team-swapped predictions. Offline inference allows future context.

For observed-ball dead/live and conditional H/A probabilities, we average the temporal network and inherited boosted predictors equally. During ball gaps, dead/live probability combines 75% temporal-network output with 25% output from a player-activity HistGradientBoosting model. Conditional H/A keeps the equal boosted/neural mixture. The activity model uses 64 player-only features; it excludes ball coordinates, gap duration, identities and other models' predictions.

We continued the parent temporal model for 3,000 steps with seed 20261116, learning rate 0.0001, AdamW weight decay 0.001 and batch size 16. We kept its training-only normalizer. H/A loss weights each training spell by `clip(sqrt(10 / duration_seconds), 1, 4)`; dead/live loss doubles on missing-ball frames. SkillCorner proxy-D weight is 0.25 and auxiliary boundary-loss weight is 0.15, with positive weight 8. The activity expert uses 676,547 sampled missing-ball training rows with equal match weights: 150 iterations, learning rate 0.05, 15 leaves, minimum leaf size 200, L2 10, 128 bins and seed 20261116.

Our decoder has H, A, dead-after-H and dead-after-A states; both dead states emit D. Minimum live duration is 0.4 s, minimum D duration 1 s, H/A transition cost 4.5 and other transition costs 9, with stride 5. We disable restart snapping. A loose ball or challenge does not automatically end live possession, and an isolated touch need not change the owning team. H–D–H counts as two possessions. We do not represent contested live possession as a separate class.

## Training data

We fitted the neural model on 65 distinct external matches. We kept the ten development matches and the 18 audit matches outside fitting and normalization.

| Source | Matches | Reference frames before augmentation | Reference construction |
|---|---:|---:|---|
| IDSSE/DFL | 7 | 1,002,644 | Provider BallStatus/BallPossession |
| Metrica samples 1 and 2 | 2 | 286,162 | Event-derived intervals |
| SkillCorner, Australian A-League | 20 | 2,955,525 | Automatic H/A phases; complementary within-period D proxy |
| PFF, 2022 World Cup | 36 | 5,636,115 | Pinned provider SDK event-state reconstruction |

The inherited boosted components use overlapping nine-game and 19-game subsets of this union. We generated 139 natural/degraded training scenarios from the same 65 matches. The activity expert uses only the 36 PFF training matches. We exclude event-informed/estimated PFF ball positions, labels and identifiers from predictor geometry and use raw visible detections.

DFL uses CC BY 4.0; SkillCorner uses MIT with attribution. Metrica references are proxies derived from its sample events. We obtained access to the original PFF release through its provider; we have not established a standard permissive license for that release and do not apply the mirror's GPL label to it. We keep provider-access qualifications and source records in `docs/PFF_SOURCE.md` and the training registries. We do not redistribute raw external or organizer data. These references have different annotation semantics and are not uniformly manual possession ground truth.

## Evaluation and model selection

We use the official Spell F1, averaged over whole matches. The selected model improved the previous development baseline on the ten PFF development games: 0.655317 to 0.680009 on original tracking and 0.642574 to 0.667271 under the fixed label-blind stress. Nine of ten games improved in each case. We reused these development games across experiments, so their scores reflect model selection.

We then compared two frozen candidates on 18 previously opened PFF World Cup matches. We changed no weights, thresholds or decoder settings during this comparison. We saved each match's predictions before opening its reference for scoring and checked old natural/stress outputs against their previous audited arrays.

| Mean official Spell F1 on 18 matches | Selected: weighted_short_gap | Alternative: uniform_temporal_focus |
|---|---:|---:|
| Original tracking | 0.639234 | 0.674644 |
| Original label-blind stress | 0.619814 | 0.655080 |
| 10 Hz, 0.5 m coordinate noise | 0.485381 | 0.479455 |
| 15 Hz, 0.5 m coordinate noise | 0.490207 | 0.484463 |
| 10 Hz, 2 m coordinate noise | 0.399510 | 0.390008 |
| 15 Hz, 2 m coordinate noise | 0.409646 | 0.397353 |

The four additional noise presets include 2–6 s ball gaps. We apply them to the canonical 25 Hz input grid; we have not calibrated them against organizer tracking errors. The alternative wins all 18 original-tracking games and 17 of 18 original-stress games, but loses under these noise presets. Its worst single-match Spell regression is 0.04885. We selected `weighted_short_gap` using the conservative regression rule written before this comparison. This prioritizes the measured noise robustness, with a clear cost on ordinary PFF tracking.

These 18 games are now part of release selection, not a fresh prospective holdout. Six scenarios still represent 18 matches. Canonical and SDK-shadow references are alternate reconstructions of the same events. The complete protocol, match list, uncertainty intervals and failed checks are in `docs/experiments/final-opened-comparison/REPORT.md`.

## Limitations

Short possessions remain a major weakness. On original tracking across the 18 games, the selected model recalls 14.23% of possessions shorter than 3 s at IoU 0.5; the alternative recalls 23.77%. Long ball gaps can cause false dead states or wrong-team assignments. The duration constraints also suppress some brief segments.

The validation games share a provider and tournament, while training sources use several proxy-label definitions. We have no independent La Liga tracking-plus-possession evaluation or hidden-match score. The results support the reported comparisons and reproducibility; they do not establish absence of overfitting or predict a finalist ranking. With more time, we would obtain a new professional-match cohort and test gap robustness and short-segment recovery before further development.

## Reproduction

Use Python 3.12, the repository's `main` branch and the original data folder containing `tactical/BAR-ATH_tactical_atd.xml` and `.txt`. The bundled frozen weights run on CPU.

```sh
git clone --branch main --single-branch https://github.com/8Tigran8/silver-octo-winner.git ruses-cules
cd ruses-cules
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_recovery_candidate.py \
  --bundle assets/recovery-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py --csv 'Ruses Cules.csv'
```

The generator refuses an existing output path. Expected CSV SHA-256:
`1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`.

All 176 regression tests pass. Clean-source CPU generations on macOS ARM, Linux ARM and emulated Linux AMD64 reproduce the same CSV and pass the official validator. Our teammate also supplied the same hash and validator output from two clean runs on a separate Linux ARM server with Python 3.12.3; we verified the reported hash and H/A/D/X counts against our file. These checks reproduce inference, not weight training or hidden-label accuracy. The manifest records source and model hashes.
