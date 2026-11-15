# Equal-provider temporal training

**RESEARCH_CANDIDATE: numeric/provider/reused-source guards pass; independent confirmation and strict inherited-fold audit pending**

This study was registered after longer training and OOD gates failed transfer checks on previously used Metrica3. It tests a change to training, not another threshold on that match. All40previously screened games are research now; no independent final test was opened.

The previous sampler gave original games1/2 and SkillCorner1/2. Since the original family has sevenDFL and twoMetrica games, Metrica received1/9of training draws. This experiment instead samples DFL/Metrica/SkillCorner each1/3, uniformly over training games/scenarios within each provider. Provider identity affects the sampler only, never inference features.

All registered fits keep the 48-channel, 25-second TCN and use 4500 fixed AdamW updates, the same loss, optimizer, normalization and augmentations as the longer baseline. Train three fixed seeds on five complete neural folds. This isolates provider sampling from the earlier longer-training family; seed trajectories change with the new sampler.

Candidate policies are three individual uniform50 fusions with v3 and their equal-probability ensemble. Select the best fixed-development candidate passing evaluation-1.0 plus the separately preregistered provider guard: primary mean spellF1 must not fall by>.01on DFL7, Metrica2 or SkillCorner20 separately. Freeze one candidate before full fitting and reused-source checks; no runner-up is chosen afterward.

| Candidate | Composite spellF1 | Gain vs v4 | Wins /29 | Numeric status |
|---|---:|---:|---:|---|
| source_seed20261113 | 0.63743 | +0.02309 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| source_ensemble | 0.63704 | +0.02269 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| source_seed20261115 | 0.63502 | +0.02067 | 24 | ELIGIBLE_FOR_CONFIRMATION |
| source_seed20261114 | 0.63414 | +0.01980 | 24 | KEEP_V4 |

## Provider guards

| Candidate | DFL gain | Metrica gain | SkillCorner gain |
|---|---:|---:|---:|
| source_ensemble | +0.02427 | +0.02321 | +0.02408 |
| source_seed20261113 | +0.02046 | +0.02854 | +0.03168 |
| source_seed20261114 | +0.01899 | +0.02941 | +0.01759 |
| source_seed20261115 | +0.01842 | +0.02714 | +0.02090 |

## Frozen reused-source checks

| Source | Scenario | V4 | Candidate | Gain |
|---|---|---:|---:|---:|
| wcsf (1) | label_blind_seed7 | 0.5702 | 0.5836 | +0.0134 |
| wcsf (1) | natural_detected | 0.6858 | 0.7090 | +0.0232 |
| metrica3 (1) | label_blind_seed7 | 0.4614 | 0.4589 | -0.0025 |
| metrica3 (1) | natural_detected | 0.5527 | 0.5630 | +0.0103 |
| european (9) | natural_detected | 0.8410 | 0.8389 | -0.0021 |

The primary composite and complete67scenario grid are unchanged. Per-game wins average two primary scenarios; augmentations are not independent games. Intervals in JSON files are descriptive, without correction for repeated selection or team/source dependence.

All labels and geometry used for fitting/scoring are public external data, never competition GT. WCSF ambiguity/anonymous identity limits, event-derived Metrica references and conditional-only European ownership remain explicit. New neural stages exclude complete folds. The unchanged legacy v3 paths exclude scored games but not every other member of that neural fold, so strict all-stage full-fold compliance and independent confirmation remain pending.

Keep verified v4 and its local submission package. The user hold bars form submission, outbound messages, push and publication. Default run.py v4 is unchanged. All failures and three seeds remain visible.
