# Temporal fusion: expanded generalization study

**Selected candidate: v4 / fusion50.** Whole-game development and external checks passed; see the separate technical reproduction receipt before claiming a packaged release.

The earlier data-only and decoder searches did not meet replacement gates. This study replaces framewise-only reasoning with a learned temporal model and evaluates source transfer. Ranking evidence below uses the unmodified official whole-spell F1; conditional European H/A scores are kept separate.

## Method and fitting

A residual dilated temporal convolutional network consumes the existing108 geometry/contact-context features plus108 missing-value masks at5Hz. Five residual blocks each contain two3-tap convolutions,48 channels, dilations1/2/4/8/16, dropout0.15. The neural receptive field is125steps (25seconds), in addition to the noncausal context already present in the geometric features. Two logits predict dead probability and home possession conditional on live play. Symmetric direct/team-swapped inference is averaged.

Normalization uses finite-feature means/stds from each training split only; missing values are zero after standardization and carry explicit masks. Standard deviations floor at0.01 and normalized values clip to[-8,8]. All period windows reset from observation/metadata-derived spans. No ownership/event field, match ID, team name or persistent player identifier is an input.

Twenty-nine public matches train the temporal model: sevenDFL, twoMetrica and twentySkillCorner. Original scenarios are matchedseed0, label-blindseed1 and clean; SkillCorner scenarios are natural and label-blindseed7. Training samples families50/50, then games/scenarios uniformly, with random team swaps. AdamW uses1500 fixed updates, batch16,256supervised steps plus64context steps each side, learning rate0.001, weight decay0.001, seed20261113 and deterministic PyTorch operations. No validation early stopping. SkillCorner dead-proxy loss has weight0.25; its live frames and original source labels weight1. Unknown targets supply no gradient.

The retained v3 probability path remains original nine-game two-stage boosting plus the nineteen-game team-only head. V4 averages its dead and conditional-team probabilities50/50 with the neural outputs, then uses the same cost9 duration-constrained decoder. The default release command loads frozen external-trained weights and runs onCPU. Neural fitting was onAppleM4MPS; repeated CPU inference with frozen weights is distinct from repeating training.

## Whole-game cross-validation

Five whole-game folds cover all29 games. Every corruption/team-swap sibling stays with its game. Each neural model is fitted without the entire scored fold (23or24training games). Baseline v3 uses its own exclusion models for the original19 and frozen nineteen-game weights for the ten games absent from its training. The original learned stack retains nested test-game exclusions. No learned baseline probabilities enter neural fitting; fusion happens only after inference. All67baseline scenario scores exactly reproduced prior results.

These are reused research games, with historically selected geometry/configuration families. Fold exclusion prevents direct estimator leakage, but does not erase historical model-selection exposure. Different training-pool sizes between methods and shared opponents are further limits. Intervals resample whole games within original/SkillCorner families, preserving paired scenarios,10000draws, seed20261113; they are descriptive, not a correction for selection or team dependence.

| Cohort | V3 spellF1 | Neural alone | Selected fusion |
|---|---:|---:|---:|
| Original matched (9) | 0.6414 | 0.6185 | 0.6440 |
| Original label-blind (9) | 0.5419 | 0.5410 | 0.5583 |
| Original stress (9) | 0.5105 | 0.5137 | 0.5320 |
| SkillCorner natural (20) | 0.5817 | 0.6662 | 0.6500 |
| SkillCorner label-blind (20) | 0.4978 | 0.6225 | 0.6051 |

Equal-quarter weighted original matched/blind and Skill natural/blind: **0.5657 → 0.6143**, gain+0.0486; descriptive95%interval [+0.0434,+0.0540]. Paired two-scenario game wins/losses: **26/3**. Registered minimum gain0.01 and no cohort mean loss>0.01 passed. Standalone neural failed the known-matched regression gate; it was not substituted after external scoring. Selection and full checkpoint were frozen in commitd0bd103 before new WCSF reference/model scores.

By primary matched/natural source, DFL changes0.6395→0.6379; Metrica0.6482→0.6651; SkillCorner0.5817→0.6500. This small DFL loss is retained in the report, not averaged away as a universal improvement.

## Overfit and provider-shift diagnostics

Compare standalone neural predictions from the full29-game fit (in-sample) with its excluded-fold predictions on the same primary scenario. Full fitting has29games while excluded fits have23/24; this is a descriptive train/CV gap, not a pure causal estimate of memorization.

| Source | Full fit in-sample spellF1 | Whole-game-out spellF1 |
|---|---:|---:|
| DFL | 0.6292 | 0.6159 |
| Metrica | 0.6498 | 0.6277 |
| SkillCorner | 0.6789 | 0.6662 |

Two further source-held-out models were fitted before scoring. DFL-only training gives SkillCorner20mean spellF1=0.5850; SkillCorner-only gives DFL7=0.5081. Mixed-source OOF neural means are0.6662 and0.6159 respectively. This shows material provider/label/sensor shift; it is not evidence that any single-source model generalizes universally. Tiny train/CV gaps alone cannot prove absence of overfit.

## Additional external checks

| Source/scenario | Metric | V3 | V4 | Scope |
|---|---|---:|---:|---|
| wcsf/label_blind_seed7 | spell_f1_mean | 0.4587 | 0.5702 | 1/1wins |
| wcsf/natural_detected | spell_f1_mean | 0.5368 | 0.6858 | 1/1wins |
| european/natural_detected | all_conditional_team_macroF1 | 0.8372 | 0.8410 | 8/9wins |
| metrica3/label_blind_seed7 | spell_f1_mean | 0.3976 | 0.4614 | 1/1wins |
| metrica3/natural_detected | spell_f1_mean | 0.5481 | 0.5527 | 1/1wins |

European9: known ownership only; unknown remainsX, notD. These games were used in the earlier v3 confirmation and are not fresh. MetricaSample3 was previously v1-scored; it never trains these models. No independent period metadata exists for its export, so predictions use the whole tracking stream rather than annotation-derived half cuts; only event-covered reference spans are scored. Event-derived labels and the whole-stream treatment limit comparability to the competition.

WCSF is a new anonymized Groningen/SciSports public research match. Its half files and clocks independently define predictor spans. Team-relative coordinate frames are reconciled from duplicate observed ball coordinates, without ownership/alive flags. Predictors never read AliveFlag, PossessionFlag or team_game_phase. Source rows at110timestamps disagree after H/A/Dmapping; only unanimous reference frames are scored, leaving266unknown25Hzframes out of143431period frames (0.1855%). Unknown GT holes affect scored spell boundaries, so this is not pristine complete ground truth. This source-format issue was resolved before any model score and without changing model weights. No majority vote or prediction-based reference repair.

Raw trajectory screening found no suspicious overlap with Metrica1/2/3 or the competition automatic tracking, across±180seconds and fouraxis reflections; this is a heuristic, not proof of anonymous identities. Only aggregate eligibility results are published. No competition public/private GT enters fitting, selection or these checks. New-source evidence is one game, so there is no meaningful across-game confidence interval for WCSF alone.

The40screened matches consist of29CVgames, nineconditional-only European games, Metrica3 and WCSF. This count must not be described as40fresh independently sampled tests. The PFF64access request and prospective36/10/12+6country-bridge split remain pending/unopened. Hidden competition quality and qualification probability are unknown.

## Reproduce

Use the repository README for frozen-weight CPU generation and two clean source-copy checks. Research fitting uses public raw inputs, optional requirements-temporal.txt and the pinned original/SkillCorner sources:

```bash
.venv/bin/python tools/temporal_trial.py prepare --jobs 3
.venv/bin/python tools/temporal_trial.py fit --device mps
.venv/bin/python tools/evaluate_temporal.py cv --device mps
.venv/bin/python tools/evaluate_temporal.py diagnostic --device mps
```

The raw-input equality audit reproduced all27original-game training feature arrays and swapped arrays exactly atfloat32. SkillCorner preparation reads pinned raw data directly. Source manifests, model hashes, complete per-match aggregate scores, exclusions, data counts and frozen plans are alongside this report. Raw inputs, predictions and intermediate arrays remain local. See [the industry/research comparison](INDUSTRY.md) for why this temporal method was selected and why a pretrained demo is not automatically a validated replacement.
