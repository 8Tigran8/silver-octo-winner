# Source-transfer and overfit diagnostics

The frozen candidate is unchanged. These are six additional neural fits: each of three providers is completely excluded, once with v4 neural training and once with the source-balanced longer training. Both models in a pair use the identical remaining training games and train-only normalization. No inherited v3 probability is mixed in, because that would reintroduce the held-out provider. These results describe the neural branch, not the full deployment pipeline.

| Held-out source | Scenario | Games | Control spellF1 | Candidate spellF1 | Gain | Wins |
|---|---|---:|---:|---:|---:|---:|
| DFL | matched_seed0 | 7 | 0.5531 | 0.5769 | +0.0238 | 7 |
| DFL | label_blind_seed1 | 7 | 0.4125 | 0.4371 | +0.0245 | 6 |
| DFL | stress_seed1 | 7 | 0.4014 | 0.4176 | +0.0162 | 6 |
| Metrica | matched_seed0 | 2 | 0.6437 | 0.7171 | +0.0735 | 2 |
| Metrica | label_blind_seed1 | 2 | 0.6347 | 0.6949 | +0.0602 | 2 |
| Metrica | stress_seed1 | 2 | 0.5863 | 0.6245 | +0.0382 | 2 |
| SkillCorner | natural_detected | 20 | 0.5035 | 0.5752 | +0.0716 | 20 |
| SkillCorner | label_blind_seed7 | 20 | 0.4599 | 0.5120 | +0.0522 | 20 |

## Training versus complete-game-out

Standalone source-balanced neural branch, same primary matched/natural scenarios. Full-fit training scores are in-sample and cannot be used as validation. The CV score uses complete excluded neural folds. The larger full training pool is an additional difference, so the gap is descriptive.

| Provider | Full-fit train spellF1 | Wholefold-out spellF1 | Train minus CV |
|---|---:|---:|---:|
| DFL | 0.6778 | 0.6576 | +0.0201 |
| Metrica | 0.8215 | 0.7013 | +0.1203 |
| SkillCorner | 0.7250 | 0.7086 | +0.0163 |

No new source was opened. References include provider-specific event/phase proxies; source semantics and optical coverage differ. Small training/CV gaps do not establish no overfit, and performance under provider holdout cannot establish hidden organizer-match quality. Every failure stays in this report; no candidate is replaced using these scores. The independent PFF confirmation remains pending.

## Remaining source sensitivity

| Provider | Whole-game-fold-out | Whole-provider-out | Difference |
|---|---:|---:|---:|
| DFL | 0.6576 | 0.5769 | -0.0807 |
| Metrica | 0.7013 | 0.7171 | +0.0159 |
| SkillCorner | 0.7086 | 0.5752 | -0.1335 |

The Metrica training/CV gap is +0.1203 on only two games: a material possible-overfit signal, despite positive provider-held-out transfer. DFL and SkillCorner also lose accuracy under whole-provider exclusion. These diagnostics do not support a no-overfit or universal-transfer claim.
