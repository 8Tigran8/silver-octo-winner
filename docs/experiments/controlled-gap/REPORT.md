# Controlled player-relative gap interpolation — rejected

The plan was committed at `3e2348e` before candidate scores. Ten existing PFF development games, two fixed observation scenarios, two predeclared gap limits and canonical/SDK-shadow references produce 80 complete records. Original final/bridge games and target labels were not used. Baseline is corrected boundary102 from `validation/observation_regression.json`, not a selectively chosen previous run.

The observed problem was false D during synthetic ball occlusion. The hypothesis was that endpoint-consistent player control supports reconstructing some missing ball positions. Both endpoints must have the same nearest track slot within 2 m; its entire path must be finite and speed ≤12 m/s. Player-relative ball offsets interpolate within each period. Only missing XY is replaced. Weights, fusion and decoder remain fixed.

| Maximum gap | Scenario | Spell F1 | Change | Positive games | Worst game change | Filled frames |
|---|---|---:|---:|---:|---:|---:|
| 2 seconds | natural | 0.655260 | −0.000057 | 6/10 | −0.011864 | 17,169 |
| 10 seconds | natural | 0.654932 | −0.000385 | 6/10 | −0.011864 | 17,627 |
| 2 seconds | blind seed 7 | 0.642574 | 0.000000 | 0/10 | 0.000000 | 43 |
| 10 seconds | blind seed 7 | 0.642574 | 0.000000 | 0/10 | 0.000000 | 43 |

Natural macro changes are −0.000525 / −0.000589; blind macro change is +0.000007 for both. None passes the predeclared +0.02 Spell F1 and 8/10 positive-games requirements. The safety guards were not relaxed after seeing scores. No candidate was promoted, no runner-up threshold sweep was made, and production inference is unchanged.

Interpretation: same-player endpoint proximity does not establish uninterrupted live possession. In the blind observation case, joint ball/player gaps also leave very few eligible frames. This experiment rejects these two bounded rules, not every possible trajectory reconstruction method. Fixing the toy scenario alone would not justify adoption.

Reproduction: `python tools/controlled_gap_study.py` after the registered external data and `tools/audit_external_regression.py --baseline validation/external_regression.json --output reports/independent_audit/observation_regression.json`. The study refuses an existing `reports/controlled_gap` directory. Two independent eligibility tests cover exact offset recovery, no mutation, changed controller, player dropout, implausible speed, overlong gaps and period edges. Aggregate results and hashes are in `result.json`; original frame-level arrays remain local.
