# Ruses Cules — More Than a Hack 2027

Current selected candidate: **weighted_short_gap**. Technical validation PASS; hidden-match quality remains uncertain. Organizer submission and repository publication are on user hold.

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

## Generate and verify the selected release

Python 3.12, frozen CPU weights, tactical ATD XML/TXT only. No training download, reference labels or research cache is required.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_recovery_candidate.py \
  --bundle assets/recovery-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'out/Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py --csv 'out/Ruses Cules.csv'
```

Expected CSV SHA-256: `1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`. Local active package: `out/selected-ready/`. Source generation commit: `dfc9e860bd28810d0aa768a584eb518e96cee0b6`; later source archive reproduction at `8b10743` gives identical bytes. The generator refuses existing output.

Read [technical details](docs/Ruses%20Cules.md), [full comparison](docs/experiments/final-opened-comparison/REPORT.md) and [active manifest](submission_manifest.json).

## Preserved alternative

`uniform_temporal_focus` remains at tag `release/frozen-uniform-temporal-focus-20261117`. Its source, assets and historical release evidence are unchanged. Local alternative CSV: `out/final-ready/Ruses Cules.csv`, SHA-256 `897db60c18fefe362acee239665136d47a8b2c87a54eb8eae67231b0dcdf9e33`. It is no longer the active upload package. Its exact gate uses `--manifest releases/uniform-temporal-focus/submission_manifest.json`. Generate it with `tools/generate_final_candidate.py --bundle assets/final-candidate/bundle.json`. The older selected model is also preserved under tag `release/rollback-weighted-short-gap-20261117`.

## Data and limits

Training uses the same 65 external games: seven DFL, two Metrica, 20 SkillCorner and 36 PFF. The activity network uses only those 36 PFF training games. Inherited boosted training registries are subsets of the same union. No development/audit match was moved into training. See [source-access qualifications](docs/PFF_SOURCE.md); original provider access is not equivalent to a standard permissive license.

Data-search priority updated on 17 November 2026 at the user's request: elite professional football, preferably La Liga, followed by other major European first divisions/Champions League and senior international tournaments. University SoccerTrack v2 is deprioritized and must not substitute for evidence of performance in elite football. The earlier frozen data review records discovery, not the current acquisition priority. PFF is the 64-match 2022 World Cup; SkillCorner is the 2024/25 Australian A-League (20 matches in the actual manifest); DFL contains two first-division and five second-division German matches from 2022/23. Metrica sample competitions are anonymized, so their level is unverified. Our ten current development matches are all PFF World Cup matches. Existing training sources are not removed without a controlled development experiment. New validation requires independent tracking/reference provenance, synchronization checks and a preregistered whole-match split. Compare tempo, possession durations, ball visibility and tracking errors instead of assuming league names establish input equivalence. No additional eligible independent La Liga tracking-plus-possession cohort has been verified; the current release remains frozen.

The current technical document uses neutral release names. Historical document hashes refer to their original versions at commit `2b88bb2`; `validation/DOCUMENTATION_UPDATE.json` records the editorial changes. Prediction bytes, weights and inference code are unchanged.
