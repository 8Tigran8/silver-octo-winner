# Local reliability and spell-aware training study

**Release: verified v4, unchanged.** New work remains local under the user hold. No organizer reference or tracking input was used.

## Registered experiments

The [original plan](plan.json) was committed before new scores/fits. Eight rule-based probability mixtures use only gap length, observed visibility or symmetric model confidence. No learned gate was fitted from globally OOF component probabilities: those would require nested outer-game exclusions to avoid indirect label leakage. The uniform50 control exactly reproduced v4 spellF1 and macroF1 in all 67 cells.

The longer family keeps architecture, sources, sampling, loss and optimizer unchanged and increases fixed updates1500 to 4500. All five neural folds were trained with each of three predeclared seeds. Three individual uniform50 fusions and four equal-probability ensemble/rule mixtures were scored.

The [spell objective](spell_plan.json) was registered after adaptive-rule failures, while long training ran and before any long-family or spell-family score. It weights short H/A runs by clipped inverse-square-root duration and valid transition neighborhoods by2, capped4. It retains the .25 SkillCorner dead-proxy multiplier. Weights affect training loss only; unknowns contribute no gradient. This family fits the same five folds and three seeds for1500updates, scoring three individual uniform50 fusions and their equal-probability ensemble.

All hypotheses, seeds, failures and selection exposure are disclosed. These are reused development games. No new independent source was opened. Engineering thresholds are the previously frozen evaluation-1.0, not post-hoc adjusted.

## Complete candidate comparison

| Candidate | Composite spellF1 | Gain vs v4 | Wins /29 | Automatic status |
|---|---:|---:|---:|---|
| long/long_seed20261115_uniform50 | 0.64454 | +0.03019 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| long/long_ensemble_uniform50 | 0.64414 | +0.02979 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| long/long_ensemble_visibility6 | 0.64328 | +0.02893 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| long/long_seed20261114_uniform50 | 0.64302 | +0.02867 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| long/long_seed20261113_uniform50 | 0.64120 | +0.02685 | 28 | ELIGIBLE_FOR_CONFIRMATION |
| long/long_ensemble_confidence | 0.64107 | +0.02672 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| long/long_ensemble_gap5 | 0.63672 | +0.02237 | 26 | ELIGIBLE_FOR_CONFIRMATION |
| spell/spell_seed20261113 | 0.63065 | +0.01630 | 23 | KEEP_V4 |
| spell/spell_seed20261115 | 0.62982 | +0.01547 | 22 | KEEP_V4 |
| spell/spell_seed20261114 | 0.62971 | +0.01536 | 24 | KEEP_V4 |
| spell/spell_ensemble | 0.62970 | +0.01535 | 24 | KEEP_V4 |
| adaptive/dead65 | 0.61885 | +0.00450 | 23 | KEEP_V4 |
| adaptive/uniform65 | 0.61780 | +0.00346 | 19 | KEEP_V4 |
| adaptive/uniform50 | 0.61435 | +0.00000 | 0 | KEEP_V4 |
| adaptive/home65 | 0.61298 | -0.00137 | 15 | KEEP_V4 |
| adaptive/visibility6 | 0.61260 | -0.00175 | 12 | KEEP_V4 |
| adaptive/confidence | 0.61173 | -0.00262 | 10 | KEEP_V4 |
| adaptive/gap10 | 0.60959 | -0.00476 | 8 | KEEP_V4 |
| adaptive/gap5 | 0.60885 | -0.00550 | 8 | KEEP_V4 |

The primary composite weights original matched/blind and SkillCorner natural/blind by one quarter each. Original stress and macro/per-class/lower-tail guards remain required. Bootstrap intervals in summary JSONs are descriptive paired whole-game intervals stratified by source family; they do not correct repeated selection or shared opponents.

## Isolation and decision limits

New neural fits exclude the entire scored fold. Inherited unchanged v3 probability paths retain scored-game exclusions as in frozen v4, not full neural-fold exclusions. This must be audited/resolved before claiming compliance with stricter all-stage fold-isolation language or promotion.

The gate checker covers metric completeness, numeric quality guards and declared neural training memberships. Its ELIGIBLE_FOR_CONFIRMATION status is not a release PASS and does not replace component-lineage review, provider-held-out checks, an unopened reserve and exact clean inference reproduction. A candidate satisfying only numeric checks cannot silently replace v4.

No public/private competition GT was scored or used to fit/choose these variants. The new predictors use public external feature caches whose raw preparation was verified in the earlier study. The original frozen PFF country split remains unopened; access/label eligibility are separate.

## Reproduce research

```bash
.venv/bin/python tools/reliability_train.py --device mps
.venv/bin/python tools/reliability_score.py adaptive
.venv/bin/python tools/reliability_score.py long --device cpu
.venv/bin/python tools/spell_train.py --device mps
.venv/bin/python tools/spell_score.py --device cpu
.venv/bin/python tools/report_reliability.py
```

The earlier pinned public-data preparation and original temporal outputs are prerequisites. Completed score files refuse overwrite; model metadata verifies folds and hashes. Research files remain under ignored reports/. See original source manifests and temporal reproduction instructions.
