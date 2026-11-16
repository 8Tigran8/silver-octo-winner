# Ruses Cules — More Than a Hack 2027

Current candidate: **weighted_short_gap**, branch **codex/short-possession-recovery** (16 November2026). Short-possession weighted training and player-activity evidence during ball gaps pass the unchanged development promotion gates. The package is reproducible; submission/publication remain on user hold. **Overall CONDITIONALLY READY**, not a guaranteed high hidden score.

| Ten original development games | Audited parent | Current | Gain | Wins |
|---|---:|---:|---:|---:|
| Natural tracking, official Spell F1 |0.655317|0.680009|+0.024692|9/10|
| Fixed blind sensor stress, official Spell F1 |0.642574|0.667271|+0.024698|9/10|

These are reused-development results with PFF event-derived references. After adoption, the fixed current model was additionally evaluated on all six opened PFF bridges and twelve old final matches; none entered training or numerical tuning during this audit. All 18 games improve in both scenarios. Earlier parent outcomes were known, so this is reopened-test evidence, not a fresh prospective holdout. Six supplementary DFL/Metrica/SkillCorner training matches remain in-sample checks.

| Later fixed-model audit | Parent Spell F1 | Current Spell F1 | Wins |
|---|---:|---:|---:|
| Six opened bridge matches, natural |0.613896|0.638797|6/6|
| Twelve opened original final matches, natural |0.604159|0.639452|12/12|
| All 18, natural |0.607404|0.639234|18/18|
| All 18, fixed blind stress |0.589488|0.619814|18/18|

Current training mean on 36 PFF matches: **0.652584**. Full game list, Netherlands–USA error analysis, leakage checks, failed reporter execution and repair: [opened-match audit](docs/experiments/recovery-opened-audit/REPORT.md). This is later evidence; the generation-bound technical document/manifest stay unchanged. The current full suite has **140 passing tests**. No absence-of-overfitting claim follows from the train/test gap.

Read the [complete experiment report](docs/experiments/spell-weighted-finetune/REPORT.md), [technical document](docs/Ruses%20Cules.md), [current audit](validation/FINAL_AUDIT_REPORT.md), and [exact manifest](submission_manifest.json). Short-possession recall remains only 15.05%; no absence-of-overfitting or finalist-probability claim is made.

## Generate the current CSV

Use Python3.12, this branch, and the original organizer data folder with tactical/BAR-ATH_tactical_atd.xml and .txt. Inference uses bundled frozen weights and CPU; no external training files, target labels, tagging or video are needed.

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

Expected SHA-256: `1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`.

All 140 tests pass. Portable inference exactly matches all 20 development outputs. Four clean-source generations (two macOS ARM, Linux ARM, emulated Linux AMD64) produce the same199,319-row CSV and pass the official validator. One physical host, no repeated training. Source commit and exact weights are recorded in the manifest. The generator refuses to overwrite existing output.

## Provenance and delivery limits

Training uses the same65 external games:7DFL,2Metrica,20SkillCorner,36 PFF. The activity expert uses only those36 PFF training games. Raw data, labels, protected organizer files and predictions remain local. See [PFF source terms](docs/PFF_SOURCE.md); original provider access is not equivalent to a standard permissive data license. The Hugging Face mirror is unused.

Current local package: out/recovery-ready/. Public repository access, source-access qualifications and team eligibility remain conditions for an unconditional READY verdict. No organizer submission occurred. The organizer deadline recorded in the welcome email is 17 November2026,23:59CET.

Historical parent documents/manifest are preserved under validation/PARENT_*. Prior six-match evidence is in [the parent audit](docs/experiments/unseen-bridge-audit/REPORT.md); earlier training and holdout results remain in docs/experiments/pff-mixed/. The old generate_mixed_candidate.py and run.py v4 still generate historical versions. Do not confuse their metrics or hashes with the current candidate.
