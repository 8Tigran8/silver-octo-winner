# Ruses Cules - team possession from tactical tracking

Selected model: **uniform_temporal_focus**. We use missing-ball temporal specialists and duration-constrained decoding to estimate H/A/D from tactical ATD XML/TXT. Technical checks pass; hidden-match accuracy and competition rank remain unknown. We submitted the selected CSV and five-page PDF on17 November2026 at19:34 CET; Google Forms confirmed receipt. The repository is public. [Submission receipt](docs/releases/submitted-20261117.json).

## Evaluation

| Mean official Spell F1 | Previous weighted_short_gap | Selected uniform_temporal_focus |
|---|---:|---:|
| Ten reused development matches, original tracking |0.680009|0.701652|
| Ten reused development matches, original blind stress |0.667271|0.686287|
| Eighteen previously opened audit matches, original tracking |0.639234|0.674644|
| Eighteen previously opened audit matches, original blind stress |0.619814|0.655080|

The selected model wins18/18 natural audit games and17/18 original-blind games. Recall of audit possessions shorter than3s increases14.23% to23.77%. These are PFF World Cup2022 games, kept outside fitting but now used for release selection. They are not a fresh prospective holdout. Source-label proxies, shared teams/provider and unresolved subsecond possessions limit generalization claims.

We initially rolled back because the selected model loses0.006–0.012 on four VV jitter/thinning/burst stress cases. We now explicitly override that conservative release choice after an unlabelled sensor-relevance review. Even stationary players gain median apparent speeds8.43–12.56m/s at0.5m jitter; the corruptor leaves all players missing on40–60% of rows. The supplied tracking has0% all-player-missing rows and median coordinate speed1.36m/s. We treat those severe combined corruptions as a robustness warning rather than a calibrated target-distribution veto. The original failed gate and selected-old decision are unchanged. See the [explicit override](docs/experiments/sensor-relevance/selection_override.json) and [sensor review](docs/experiments/sensor-relevance/REPORT.md). This post-result choice adds selection bias and does not guarantee hidden quality.

## Generate and verify

Use Python3.12 and the main branch. Frozen weights are bundled; inference requires only the original tactical data, no training download or research cache.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_final_candidate.py \
  --bundle assets/final-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'out/Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py --csv 'out/Ruses Cules.csv'
```

The generator refuses existing output. Expected CSV SHA-256:
`897db60c18fefe362acee239665136d47a8b2c87a54eb8eae67231b0dcdf9e33`.

**Current local upload package: out/submission-final/Ruses Cules.csv and out/submission-final/Ruses Cules.pdf.** The older out/selected-ready/ directory is a preserved rollback, not the active package. [PDF technical report](docs/Ruses%20Cules.pdf), [Markdown source](docs/Ruses%20Cules.md), [active manifest](submission_manifest.json).

All176 regression tests pass. Four clean-source generations across macOS ARM, offline Linux ARM and emulated Linux AMD64 reproduced the same CSV on one physical host. VV separately reports two matching Linux ARM server runs of this model: [reported evidence](docs/experiments/final-opened-comparison/vv_external_reproduction.json). These reproduce frozen-weight inference, not repeated training or target-label accuracy. The previous model's later detailed server receipt remains separately recorded in validation/EXTERNAL_REPRODUCTION.json.

## Training sources and limits

Training uses65 unique external matches: seven DFL (two Bundesliga and five Bundesliga2), two anonymized Metrica sample matches,20 SkillCorner A-League2024/25 matches and36 PFF World Cup2022 games. Inherited boosted registries overlap the same union. No development or audit match was added to fitting. Raw PFF predictor geometry excludes event-informed/estimated coordinates. See training registries and [PFF access qualifications](docs/PFF_SOURCE.md). Original provider access does not establish a standard permissive redistribution license; raw data are excluded.

We prioritize elite professional football and independent La Liga tracking-plus-possession validation. Such an additional eligible cohort has not been verified. University data do not substitute for evidence in elite football. The current release remains frozen; further improvement requires a new whole-match protocol and source/annotation review.

## Rollback and records

The previous weighted model, weights, CSV and manifest remain intact. Its post-comparison manifest is releases/weighted-short-gap/post-comparison-manifest.json. Original comparison plan/results/failed gates are under docs/experiments/final-opened-comparison/. Historical document versions and neutral reference aliases are recorded in validation/DOCUMENTATION_UPDATE.json. No history was rewritten.

To regenerate the report, install requirements-report.txt and run tools/build_submission_pdf.py. Model inference does not require report-generation dependencies. The PDF is the upload document; Markdown remains available for inspection.
