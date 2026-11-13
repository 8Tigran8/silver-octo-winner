# Fixed-capacity 19 → 29 game study

**Verified release remains v3.** No fresh confirmation has been completed in this cycle.

The only experimental change is adding ten prior SkillCorner confirmation games to the team-head training pool. They are now training data, not a test set. The expanded head retains 108 features, original HGB capacity, augmentations, sampling and a fixed 50/50 blend with the original nine-game stacked team probabilities. The dead-probability channel and decoder stay fixed.

The 29-game head uses 2,311,272 unique live-team source frames and 1,155,592 sampled augmented/symmetric examples, versus 747,644 examples for the nineteen-game head. Provider frame counts are not independent statistical samples.

One full model and nineteen leave-one-whole-game-out heads are fitted. Every augmented and team-swapped sibling of a scored game is excluded from its head. Both v3 and data29 are evaluated on the same original19 excluded games; baseline scores match the prior study exactly. Historical feature/configuration choices reused these games, so this is reused-development evidence, not an unbiased final generalization estimate.

| Cohort | V3 spellF1 | Data29 spellF1 | Difference |
|---|---:|---:|---:|
| known/matched_seed0 | 0.6414 | 0.6415 | +0.0000 |
| known/label_blind_seed1 | 0.5419 | 0.5436 | +0.0016 |
| known/stress_seed1 | 0.5105 | 0.5110 | +0.0005 |
| development/natural_detected | 0.5589 | 0.5601 | +0.0013 |
| development/label_blind_seed7 | 0.4800 | 0.4803 | +0.0003 |

Weighted gain: **+0.0007**; descriptive paired 95% game-bootstrap interval [-0.0015, +0.0029]. Fixed development gate pass: **False**. Gate requires gain ≥0.005 and no cohort mean loss >0.005. Original and SkillCorner games are resampled separately, preserving paired scenario differences within each game; 10,000 draws, seed 20261113. Intervals do not account for historical model selection.

## Descriptive overfit diagnostic

For each excluded fit, score at most 2,048 evenly spaced augmented/symmetric H/A rows per game. Training means weight games equally and include only the same source family as the held-out cohort; held-out rows use the same augmentation pipeline. These binary-head losses/accuracies are not the official decoded spellF1. A train/held-out gap can indicate overfit or match shift; a small gap cannot prove absence of overfit.

| Held-out family | Data29 training loss | Data29 held-out loss | V3 held-out loss | Data29 training accuracy | Data29 held-out accuracy | V3 held-out accuracy |
|---|---:|---:|---:|---:|---:|---:|
| original_pool | 0.2024 | 0.2361 | 0.2366 | 0.9126 | 0.8942 | 0.8930 |
| SkillCorner | 0.1725 | 0.2173 | 0.2244 | 0.9285 | 0.9073 | 0.9023 |
| IDSSE_DFL | 0.2240 | 0.2446 | 0.2433 | 0.9013 | 0.8910 | 0.8905 |
| Metrica | 0.1327 | 0.2063 | 0.2129 | 0.9473 | 0.9053 | 0.9019 |

## Release and further data

The prior European9 have also been scored already; they cannot become fresh again. A new third-provider confirmation is required before promoting a replacement. PFF officially released 64 World Cup2022 games, but access to the OpenSportsLab mirror requires login, contact-sharing consent and owner approval. The access request was submitted with user consent and is pending owner approval. No gated file or organizer match material enters this experiment. No competition public/private GT is read. The verified v3 CSV/package remains unchanged.

Plan: [plan.json](plan.json). Full aggregate scores, game exclusions, input-row/model hashes and diagnostics are alongside this report. [PFF split](pff_protocol.json), [access status](pff_access_status.json), [source eligibility](pff_source_eligibility.json) and [fixture deduplication](fixture_identity_check.json) are also recorded. Raw data/weights/arrays remain local. Original source references and licences are documented in the v3 method.

Reproduce with pinned original/SkillCorner sources and prior baseline feature/probability caches:

```bash
.venv/bin/python tools/more_data_trial.py prepare --jobs 4
.venv/bin/python tools/more_data_trial.py fit --jobs 2
.venv/bin/python tools/more_data_trial.py score --jobs 4
.venv/bin/python tools/report_more_data.py
```

## Bounded follow-up: temporal costs and conditional calibration

After the data-only result, nine decoding policies were registered before scoring. All reuse fixed29-game weights, the same whole-game exclusions and the same known development games. This is further adaptive development, not additional independent validation. The c9_t1 control reproduces the initial data29 scores exactly.

| Policy | Known matched | Known label-blind | Known stress | Skill natural | Skill label-blind | Weighted gain vs v3 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| c15_t0.75 | 0.6403 | 0.5449 | 0.5141 | 0.5680 | 0.4909 | +0.0042 | False |
| c12_t1 | 0.6392 | 0.5430 | 0.5115 | 0.5683 | 0.4916 | +0.0036 | False |
| c15_t1 | 0.6335 | 0.5388 | 0.5078 | 0.5712 | 0.4979 | +0.0023 | False |
| c12_t0.75 | 0.6417 | 0.5456 | 0.5140 | 0.5617 | 0.4832 | +0.0020 | False |
| c9_t1.25 | 0.6401 | 0.5397 | 0.5086 | 0.5664 | 0.4847 | +0.0015 | False |
| c12_t1.25 | 0.6338 | 0.5361 | 0.5062 | 0.5696 | 0.4952 | +0.0010 | False |
| c9_t1 | 0.6415 | 0.5436 | 0.5110 | 0.5601 | 0.4803 | +0.0007 | False |
| c9_t0.75 | 0.6439 | 0.5454 | 0.5122 | 0.5536 | 0.4757 | -0.0002 | False |
| c15_t1.25 | 0.6253 | 0.5324 | 0.5032 | 0.5688 | 0.5007 | -0.0022 | False |

Selected research candidate: **none**. V3 remains the verified release. No European9 or PFF confirmation scores are opened for this sweep.
