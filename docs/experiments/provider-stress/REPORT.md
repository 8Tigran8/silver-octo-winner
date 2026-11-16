# Supplemental provider/sensor stress diagnostic

16 November 2026. **Robustness weakness found. No model changes.**

VV's [PR #2](https://github.com/8Tigran8/silver-octo-winner/pull/2), commit `d69ee6b900efbb4dd45c626e5b2a3c2578e57d1c`, contributes the converter, four presets and twelve synthetic correctness tests. It is integrated into the local research branch; remote main and the organizer submission remain untouched.

## Result and interpretation

Boundary102 improves both canonical/native controls, but loses all **six distinct perturbed conditions**, by -0.06944 to -0.00638 spellF1 against same-exclusion v4. Eight named perturbed copies reduce to six distinct inputs because native 10-Hz SkillCorner cannot produce a separate 15-Hz measurement stream. These are only **two familiar development games**, not six/eight new independent matches.

This is negative evidence about synthetic cadence/noise robustness. It does not overwrite the independently frozen 12-game PFF result, but it prevents a claim that the new model dominates v4 under every sensor condition. The main candidate, mixture, decoder and submitted-file candidate are unchanged; no seed, condition, game or losing row is dropped. Original development/final gates passed; this supplemental diagnostic was explicitly not registered as a new promotion gate.

## Complete combined-pilot results

| Source / fold | Condition | V4 spellF1 | Boundary102 | Gain | MacroF1 gain | Ball missing | Duplicate of |
|---|---|---:|---:|---:|---:|---:|---|
| original/idsse_J03WMX / 2 | baseline | 0.62955 | 0.64970 | +0.02015 | +0.00328 | 0.2% | - |
| original/idsse_J03WMX / 2 | native_observed_only | 0.62955 | 0.64970 | +0.02015 | +0.00328 | 0.2% | - |
| original/idsse_J03WMX / 2 | hz10_noise0p5 | 0.33846 | 0.33209 | -0.00638 | +0.07907 | 64.7% | - |
| original/idsse_J03WMX / 2 | hz15_noise0p5 | 0.35649 | 0.29931 | -0.05718 | -0.00203 | 47.1% | - |
| original/idsse_J03WMX / 2 | hz10_noise2 | 0.31307 | 0.28273 | -0.03034 | +0.00294 | 64.7% | - |
| original/idsse_J03WMX / 2 | hz15_noise2 | 0.33257 | 0.26313 | -0.06944 | -0.00593 | 47.1% | - |
| skillcorner/1925299 / 4 | baseline | 0.62239 | 0.63750 | +0.01511 | +0.01440 | 37.3% | - |
| skillcorner/1925299 / 4 | native_observed_only | 0.63825 | 0.65768 | +0.01943 | +0.01234 | 74.9% | - |
| skillcorner/1925299 / 4 | hz10_noise0p5 | 0.51429 | 0.48242 | -0.03187 | +0.01150 | 78.0% | - |
| skillcorner/1925299 / 4 | hz15_noise0p5 | 0.51429 | 0.48242 | -0.03187 | +0.01150 | 78.0% | hz10_noise0p5 |
| skillcorner/1925299 / 4 | hz10_noise2 | 0.41922 | 0.39099 | -0.02823 | -0.00107 | 78.0% | - |
| skillcorner/1925299 / 4 | hz15_noise2 | 0.41922 | 0.39099 | -0.02823 | -0.00107 | 78.0% | hz10_noise2 |

`baseline` retains the exact existing reader format. `native_observed_only` uses the source-native frequency, zero noise and zero dropout bursts. For SkillCorner, baseline includes 10-to-25-Hz interpolation while all observed-only/stress inputs leave unmapped grid rows NaN. Therefore baseline-to-stress degradation combines representation, cadence, coordinate noise and bursts. The extra native control separates representation from added corruption.

Noise is independent zero-mean Gaussian noise per x/y coordinate (0.5 m or 2 m), applied to ball and players. Ball burst starts follow a two-per-minute Poisson process, with uniform 2-6-second duration. No holding, interpolation, clipping or temporal correlation is introduced by the stress converter. Source gaps and half boundaries remain intact. Such framewise jitter is a synthetic failure probe, not a calibrated estimate of real tracking errors.

## Post-hoc one-factor decomposition

These explanatory probes were registered after seeing combined-pilot losses. They are descriptive, use the same frozen models/folds, and do not constitute new confirmation or a tuning/adoption rule.

| Source | One changed factor | V4 spellF1 | Boundary102 | Gain |
|---|---|---:|---:|---:|
| original/idsse_J03WMX | noise_only_0.5 | 0.37992 | 0.31902 | -0.06091 |
| original/idsse_J03WMX | noise_only_2.0 | 0.35573 | 0.28206 | -0.07367 |
| original/idsse_J03WMX | burst_only | 0.61931 | 0.64185 | +0.02254 |
| original/idsse_J03WMX | rate_only_10 | 0.58756 | 0.59451 | +0.00695 |
| original/idsse_J03WMX | rate_only_15 | 0.60816 | 0.64289 | +0.03473 |
| skillcorner/1925299 | noise_only_0.5 | 0.52687 | 0.52325 | -0.00361 |
| skillcorner/1925299 | noise_only_2.0 | 0.43871 | 0.39396 | -0.04475 |
| skillcorner/1925299 | burst_only | 0.60880 | 0.63390 | +0.02510 |

## Isolation and reproducibility

All predictions use the original game's excluded outer-fold models (DFL fold 2; SkillCorner fold 4). Every inherited direct/nested stage excludes the whole scored fold. Both old and new neural normalizers are exactly recomputed from their training-only rows. Full-fit release weights are never used for these familiar games. The original natural SkillCorner scores exactly reproduce the previously recorded spell/macro/class/spell-count metrics for both models.

Only geometry, clocks, team-slot mapping, pitch size and independently supplied periods reach prediction. Reference arrays are attached after corruption and enter scoring after both predictions exist. Twelve converter tests verify reference independence, missingness, sampling clocks and period boundaries. The original 67 frozen model/source/evidence files pass before/after hash checks. No target-match input, organizer GT or original PFF final reference is scored in this diagnostic.

The runners stream one full-match geometry condition at a time and save only aggregate receipts and ignored local predictions, avoiding the standalone exporter's 20-GiB capacity requirement. Raw tracking and reference arrays are not redistributed.

```bash
.venv/bin/python -m unittest tests.test_provider_stress -v
.venv/bin/python tools/evaluate_provider_stress.py
.venv/bin/python tools/diagnose_provider_stress.py
python3 tools/report_provider_stress.py
```

Completed diagnostic endpoints refuse overwrite. Full research reproduction additionally requires the original public raw-data files, cached training-only normalizers and audited excluded-fold artifacts, as described in prior studies. Release inference remains the separate frozen-bundle command and requires none of these research caches.

## Consequence for the current solution

Keep boundary102 as the selected candidate on the strength of the untouched PFF final, while disclosing this additional weakness and the Metrica overfit risk. No model is switched using the consumed final. Improving jitter robustness is future research requiring development-only fitting and a new untouched reserve before replacement. No organizer submission or remote publication is performed by these scripts.
