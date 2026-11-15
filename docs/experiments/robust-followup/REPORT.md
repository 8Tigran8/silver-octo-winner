# Reference-blind temporal robustness followup

NOT_ADOPTED: reused-source guard failed

The earlier long-TCN candidate gained+0.03019on29reused development games, but then regressed on Metrica3. This followup was registered after that observed transfer failure and before new rule scores. These are further development experiments; no source is relabelled fresh.

Fourteen configurations include old/long controls, channel-specific old/new neural mixtures, temperature softening and unsupervised outlier gates. The feature gate averages the eight largest absolute training-normalized observed features, then uses a period-reset five-second mean. Means/scales come from the excluded-fold training data only. No ownership/alive/event target influences inference weights.

Select the highest-scoring changed rule passing evaluation-1.0 on the fixed67cells; old/long controls are not new corrective candidates. The selected OOD4to8rule was frozen before checking other seeds and repeating old-source checks. No runner-up was chosen from those checks. Both control metrics reproduce their old scores exactly.

| Candidate | Composite | Gain | Wins /29 | Numeric dev status |
|---|---:|---:|---:|---|
| robust_long_control | 0.64454 | +0.03019 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_ood4to8 | 0.64433 | +0.02998 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_ood3to6 | 0.64329 | +0.02895 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_ood2to4 | 0.64062 | +0.02627 | 28 | ELIGIBLE_FOR_CONFIRMATION |
| robust_temperature1.5 | 0.63960 | +0.02525 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_new_dead_half_home | 0.63841 | +0.02407 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_alpha75 | 0.63792 | +0.02357 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_new_home_half_dead | 0.63590 | +0.02156 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_alpha67 | 0.63564 | +0.02129 | 29 | ELIGIBLE_FOR_CONFIRMATION |
| robust_temperature2.0 | 0.63257 | +0.01822 | 29 | KEEP_V4 |
| robust_alpha50 | 0.63002 | +0.01567 | 29 | KEEP_V4 |
| robust_old_home | 0.63002 | +0.01567 | 28 | KEEP_V4 |
| robust_old_dead | 0.62826 | +0.01391 | 24 | KEEP_V4 |
| robust_old_control | 0.61435 | +0.00000 | 0 | KEEP_V4 |

## Frozen reused-source checks

| Source | Scenario | V4 | Candidate | Gain |
|---|---|---:|---:|---:|
| wcsf (1) | label_blind_seed7 | 0.5702 | 0.5829 | +0.0127 |
| wcsf (1) | natural_detected | 0.6858 | 0.7135 | +0.0277 |
| metrica3 (1) | label_blind_seed7 | 0.4614 | 0.4203 | -0.0410 |
| metrica3 (1) | natural_detected | 0.5527 | 0.5289 | -0.0238 |
| european (9) | natural_detected | 0.8410 | 0.8404 | -0.0005 |

All eleven games had previously been used, and Metrica3 failure motivated the hypothesis. These are development/regression evidence, not independent final confirmation. WCSF ambiguity/identity limits, event-derived Metrica labels and conditional-only European ownership labels remain unchanged. Other fixed seeds and all failed candidates are published as aggregate files.

The 29/29 game win count uses each game mean across its two primary scenarios. Extra corruptions are not independent matches. Bootstrap intervals are descriptive and do not correct repeated selection, opponent dependence or source effects.

Legacy component isolation still excludes the scored game rather than every other game in its neural fold. New TCN fitting excludes the whole fold. See ../reliability/lineage_audit.json; do not claim strict all-stage full-fold compliance. No competition GT was accessed. Only public external games trained the weights and supplied quality references; separate local file generation may use organizer automatic tracking without scoring its GT.

Verified v4 and its local submission package remain unchanged. The user hold bars submissions, messages, pushes and publication. No independent reserve was opened; final qualification and hidden-match quality are unknown.
