# Current release audit

## Final frozen comparison, 17 November 2026

The active release is **weighted_short_gap**, selected by the conservative rule registered before this audit. The alternative **uniform_temporal_focus** remains intact. This is a robustness-policy rollback, not a claim that the old model performs better on ordinary PFF tracking.

| Mean official Spell F1, 18 previously opened PFF World Cup matches | Selected weighted_short_gap | Alternative uniform_temporal_focus |
|---|---:|---:|
| Original tracking | 0.639234 | 0.674644 |
| Original label-blind stress | 0.619814 | 0.655080 |
| VV 10 Hz, noise 0.5 m | 0.485381 | 0.479455 |
| VV 15 Hz, noise 0.5 m | 0.490207 | 0.484463 |
| VV 10 Hz, noise 2 m | 0.399510 | 0.390008 |
| VV 15 Hz, noise 2 m | 0.409646 | 0.397353 |

The alternative wins all 18 original-tracking matches and 17/18 original-stress matches. Original-tracking short-possession recall (<3 s) increases 14.23% to 23.77%. It nevertheless fails the preregistered mean/macro/worst-match guards under VV corruption. The largest single-match Spell regression is 0.04885. All four VV presets include 2–6-second ball gaps. Their noise is applied to the audited 25 Hz model-input grid; equivalence to the organizer sensor is unproven. This conservative veto is an engineering policy, not evidence that the selected model maximizes hidden-match quality.

All 18 matches were previously opened, although none entered fitting or normalization. This comparison now also uses them for release selection. Six scenarios and two reference reconstructions remain 18 matches, not independent datasets. No fitting, threshold search, result-dependent exclusion or target-label scoring occurred. Predictions were saved before references were opened for scoring. Canonical and SDK-shadow references matched their previous audited arrays; old natural/blind predictions matched their previous audit exactly. The audit blocked access to organizer data/labels.

All 176 regression tests pass. The selected model was reproduced again from clean source archive 8b1074340e2734f34e4cbc58930fb4bd2c309df7, exposing only the two tactical ATD files. The 199,319-row CSV matches the historical artifact byte-for-byte and passes the official validator and exact source/model gate. This supplements the historical four clean-source inference reproductions on one physical host; it does not repeat model training. VV supplied the same SHA-256 and official validator output from two clean runs of this selected release on a separate Linux ARM server (Python 3.12.3). We checked the hash, row count and H/A/D/X counts against the selected CSV; the server logs were supplied by our teammate.

Full protocol, every match, diagnostics, failures and uncertainty: docs/experiments/final-opened-comparison/REPORT.md. Replay the comparison from audit snapshot bf30915, where the root manifest still matches freeze.json; later root-manifest changes only activate the selected package. A frozen alternative manifest is preserved at releases/uniform-temporal-focus/submission_manifest.json. No absence-of-overfitting, La Liga generalization, hidden score or finalist probability is certified. Submission and repository publication remain on user hold.

Active manifest: submission_manifest.json. Active local package: out/selected-ready/. Historical release evidence remains under releases/weighted-short-gap/ and docs/releases/uniform-temporal-focus/.
