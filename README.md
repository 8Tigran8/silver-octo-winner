# Ruses Cules — More Than a Hack 2027

Current candidate: **weighted_short_gap**, branch **codex/short-possession-recovery** (16 November2026). Short-possession weighted training and player-activity evidence during ball gaps pass the unchanged development promotion gates. The package is reproducible; submission/publication remain on user hold. **Overall CONDITIONALLY READY**, not a guaranteed high hidden score.

| Ten original development games | Audited parent | Current | Gain | Wins |
|---|---:|---:|---:|---:|
| Natural tracking, official Spell F1 |0.655317|0.680009|+0.024692|9/10|
| Fixed blind sensor stress, official Spell F1 |0.642574|0.667271|+0.024698|9/10|

These are reused-development results with PFF event-derived references. The six recently opened PFF bridges and twelve old final matches were not used to develop or score this change. Their old results belong to boundary102 only. Six supplementary DFL/Metrica/SkillCorner **training** matches test provider regressions; they are not a new test set.

Read the [complete experiment report](docs/experiments/spell-weighted-finetune/REPORT.md), [technical document](docs/Ruses%20Cules.md), [current audit](validation/FINAL_AUDIT_REPORT.md), and [exact manifest](submission_manifest.json). Short-possession recall remains only15.05%; no absence-of-overfitting or finalist-probability claim is made.

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

All136 tests pass. Portable inference exactly matches all20 development outputs. Four clean-source generations (two macOS ARM, Linux ARM, emulated Linux AMD64) produce the same199,319-row CSV and pass the official validator. One physical host, no repeated training. Source commit and exact weights are recorded in the manifest. The generator refuses to overwrite existing output.

## Provenance and delivery limits

Training uses the same65 external games:7DFL,2Metrica,20SkillCorner,36PFF. The activity expert uses only those36PFF training games. Raw data, labels, protected organizer files and predictions remain local. See [PFF source terms](docs/PFF_SOURCE.md); original provider access is not equivalent to a standard permissive data license. The Hugging Face mirror is unused.

Current local package: out/recovery-ready/. Public repository access, source-access qualifications and team eligibility remain conditions for an unconditional READY verdict. No organizer submission occurred. The organizer deadline recorded in the welcome email is17 November2026,23:59CET.

Historical parent documents/manifest are preserved under validation/PARENT_*. Prior six-match evidence is in [the parent audit](docs/experiments/unseen-bridge-audit/REPORT.md); earlier training and holdout results remain in docs/experiments/pff-mixed/. The old generate_mixed_candidate.py and run.py v4 still generate historical versions. Do not confuse their metrics or hashes with the current candidate.
