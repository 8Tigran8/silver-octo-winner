# Short-possession and missing-ball improvement — completed

16 November 2026. **Adopted: weighted_short_gap.** This is a measured development improvement, not a new independent-test result or proof against overfitting. No organizer submission or repository visibility change was made.

## Scope and decision

All ten original PFF development matches were used, with natural detections and the previously fixed label-blind sensor stress. References are provider-event-derived H/A/D proxies, checked against both canonical and SDK-shadow reconstructions; both give identical results. The six recently opened bridge fixtures, original twelve final fixtures and organizer target labels were not used for this round's fitting, comparison, selection or calibration. Their historical scores belong to the parent model only.

The original acceptance rule was kept: each scenario and reference must improve mean official Spell F1 by at least0.020, improve at least8/10 games, lose at most0.020 on any game, and lose at most0.002 mean macro F1. Both composed policies pass. The registered rule chooses the largest minimum gain across scenarios: weighted_short_gap. Selection was frozen before supplementary provider checks or target CSV generation. No settings were changed afterward.

The complete round evaluated22 changed policies plus the unchanged parent. This is an adaptive development sequence: each later hypothesis was registered before its own measurements, after seeing earlier development results. It is not one-shot hypothesis testing. Three new fits were made: player-activity HGB, uniform neural continuation, weighted neural continuation. Failed policies and their original gates remain recorded in the linked short-gap study.

## Hypotheses, experiments and decisions

1. Relaxing minimum live duration alone does not produce a meaningful gain: brief-control evidence is often missing before decoding. Keep no standalone decoder change.
2. Player-relative gap bridging and measured-contact corrections fail the real-match gate; contact creates harmful extra switches. Neither is in the selected model.
3. A player-activity head partly complements neural dead-ball estimates during missing-ball intervals. Its composition with the original network improves both scenarios but falls below the0.020 gate. Retain only as a research component for a separately registered composition test.
4. Uniform frame loss underweights short possessions. Continue the same parent network for3000 steps with training-only short-spell and missing-ball loss weights. A matched uniform-loss continuation uses the same regenerated139 training scenarios, seed, optimizer, steps, model and train-only normalizer. This separates weighting from merely doing more training. Both improve, but neither standalone policy passes the gate.
5. Compose the weighted network with the already fitted activity expert and the already specified decoder choices. No new fit, blend search or threshold tuning. Both compositions pass; the frozen selection rule picks weighted_short_gap.

For the unchanged decoder, weighting adds +0.005875 natural Spell F1 versus matched uniform continuation (10/10 wins), and +0.003372 blind (6/10 wins). Therefore not all the final gain is attributable to weighting alone.

## Complete neural/composition outcomes

| Policy | Scenario | Spell F1 | Gain vs parent | Wins | Worst game gain | Original gate |
|---|---|---:|---:|---:|---:|---|
| uniform | natural_detected | 0.662963 | +0.007646 | 10/10 | +0.001044 | FAIL |
| uniform | label_blind_seed7 | 0.651812 | +0.009239 | 10/10 | +0.000306 | FAIL |
| uniform_short | natural_detected | 0.664533 | +0.009217 | 7/10 | -0.005101 | FAIL |
| uniform_short | label_blind_seed7 | 0.653405 | +0.010831 | 9/10 | -0.004104 | FAIL |
| weighted | natural_detected | 0.668838 | +0.013521 | 10/10 | +0.002248 | FAIL |
| weighted | label_blind_seed7 | 0.655184 | +0.012611 | 10/10 | +0.000329 | FAIL |
| weighted_short | natural_detected | 0.670783 | +0.015467 | 10/10 | +0.002248 | FAIL |
| weighted_short | label_blind_seed7 | 0.655412 | +0.012838 | 10/10 | +0.000132 | FAIL |
| weighted_gap | natural_detected | 0.678751 | +0.023434 | 9/10 | -0.003055 | PASS |
| weighted_gap | label_blind_seed7 | 0.668110 | +0.025536 | 10/10 | +0.001667 | PASS |
| weighted_short_gap | natural_detected | 0.680009 | +0.024692 | 9/10 | -0.002849 | PASS |
| weighted_short_gap | label_blind_seed7 | 0.667271 | +0.024698 | 9/10 | -0.000622 | PASS |

## Selected model error analysis

| Development measure | Parent natural | Selected natural | Parent blind | Selected blind |
|---|---:|---:|---:|---:|
| Mean official Spell F1 |0.655317|0.680009|0.642574|0.667271|
| Mean macro F1 |0.885608|0.895472|0.877913|0.892163|
| Short H/A recall, duration<3s, IoU0.5 |65/578 (11.25%)|87/578 (15.05%)|55/578 (9.52%)|76/578 (13.15%)|
| False D among missing-ball reference-live frames |71319/339698 (20.99%)|67196/339698 (19.78%)|84256/380531 (22.14%)|69574/380531 (18.28%)|
| False live among missing-ball reference-D frames |23715/536842 (4.42%)|15346/536842 (2.86%)|22767/540597 (4.21%)|17105/540597 (3.16%)|

Nine of ten games improve in each scenario. Worst natural loss is0.002849; worst blind loss0.000622. Short-spell recall is only a diagnostic: it ignores false-positive precision and uses one threshold, unlike official Spell F1. At15.05%, short possession recovery remains a substantial weakness despite the improvement.

Descriptive paired-match bootstrap95% gain ranges are[0.014807,0.034286] natural and[0.016045,0.032698] blind. Reused development, shared countries/provider and model selection invalidate interpreting these as calibrated generalization guarantees.

## Supplementary checks and packaging

After selection, six preregistered DFL/Metrica/SkillCorner **training** games were checked in two scenarios each. All12 cells improve and all unchanged provider/worst-game/macro guards pass. These six are different from the untouched PFF bridges and are in-sample regression checks, not additional independent validation. Mean Spell gains: DFL+0.027809 matched/+0.037838 blind; Metrica+0.041479/+0.035471; SkillCorner+0.020201 natural/+0.030179 blind. See provider_regression.json.

The portable package matches every predicted frame on all20 development game/scenario outputs (portable_parity.json). All136 tests pass with no skips, failures or errors, plus60 brute-force decoder comparisons. Tests cover period isolation, team swapping, finite output, incomplete coordinates, default-pitch handling, loss masking and unchanged uniform loss. Unit tests establish contracts, not match quality.

Four fresh-source generations (two macOS ARM, one Linux ARM, one emulated Linux AMD64) produce byte-identical199,319-row CSVs and pass the official validator. Linux network is disabled; only the original two tracking files are mounted. One physical host/operator was used, and training was not repeated. Source commit: dfc9e860bd28810d0aa768a584eb518e96cee0b6. CSV SHA-256: `1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`.

## Reproduce and evidence

Plans, freezes, complete score records, fit provenance and selection are in this directory. tools/spell_weighted_study.py implements matched fitting/scoring; tools/combined_recovery_study.py implements the frozen composition; tools/recovery_provider_regression.py implements supplementary checks. They require original externally obtained training data and local prepared inputs. The portable production generator needs only bundled weights and the organizer tracking files, not research caches or labels. See the current technical document for commands.

selection.json deliberately preserves production_adopted=false at the earlier selection freeze. adoption.json records the later verified adoption; no frozen evidence was rewritten. The preceding failed policies remain in ../short-gap-recovery/REPORT.md. Historical holdout scores are preserved in validation/PARENT_FINAL_AUDIT_REPORT.md and must not be attributed to this new model.

Further development-only searches would increase selection bias. The current changes meet the meaningful-improvement and safety gates. Fresh provider-diverse labels remain the main missing evidence for the new model's generalization; no absence-of-overfitting or finalist-probability claim is made.
