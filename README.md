# Ruses Cules · More Than a Hack 2027

Shared team repository for the Phase 1 ball-possession challenge.

Official challenge and evaluation code: [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027).

## Submission contract

Produce a UTF-8 CSV with header `frame,label`, exactly **199,319** rows and ordered frames **0..199318**. Labels: `H` (home possession), `A` (away possession), `D` (dead ball), `X` (abstain). The ranking metric is **spellF1**; macroF1 is the first tie-break.

The team submission also needs this public source repository and a short technical document covering the approach, assets, training data, possession rules, limitations and reproduction command.

Deadline from the organizer's welcome email: **17 November 2026, 23:59 CET**.

## Install and prepare

Use Python 3.12. Run the commands from the repository root. Full external source files take about 2.75 GB; conversions, features and training need additional disk space. Start with two workers on a machine with 16 GB RAM.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-validation.txt
.venv/bin/python tools/fetch_external.py
.venv/bin/python tools/ext_idsse.py
.venv/bin/python tools/ext_metrica.py
.venv/bin/python tools/fetch_reference.py
```

`tools/external_sources.json` pins the public training source revisions, filenames, sizes and object hashes. The downloader verifies them. IDSSE is distributed under CC BY 4.0; Metrica sample-data asks users to acknowledge its source. See the source links in the manifest and [the technical draft](docs/Ruses%20Cules.md).

Obtain the competition files from the organizer's private download link and store them locally:

```text
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
```

The v1 method uses these two files. It does not read the video, homography or Smart Tagging. Set `MTAH_DATA_DIR` to use a different data directory.

## Generate the team CSV

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python run.py v2 --jobs 2 --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
```

This retrains the original learned weights on the nine external matches and runs full-match inference with the externally selected v2 decoder. `run.py v1` retains the original cost-6 decoder; v2 uses cost 9. Features, learned model settings and duration constraints are unchanged. It writes exactly one CSV and checks the local CSV contract. The second command uses the unmodified official validator pinned to `f8f2629ff1e0985f81d7e59d8413678514ee1530`. A format PASS does not measure prediction quality.

Prediction does not load or score the public ground truth. An explicit `run.py score ...` command remains available for a pipeline correctness check; do not use it to select models or tune parameters.

## Verify from two clean source copies

Commit the intended source changes, then run:

```bash
.venv/bin/python tools/reproduce.py --data-dir /absolute/path/to/data --method v2 --jobs 2
```

The verifier exports the committed source twice, independently converts the pinned raw external matches, retrains the model in each copy, validates both CSVs with the official validator, compares their SHA-256 hashes and checks that the input files did not change. CSVs, logs and a receipt stay under `reports/reproduction/`. It does not evaluate the public window. Both runs use the installed pinned environment; this does not establish reproducibility across operating systems or library builds.

For decoder and scorer checks on synthetic fixtures and the organizer's published baselines:

```bash
.venv/bin/python -m tests.test_boost_viterbi
.venv/bin/python -m tests.test_metrics
```

## External quality evidence

The fixed v1 pipeline achieved mean **spellF1 0.6466 versus 0.3244** for the organizer baseline on nine external matches using nested match cross-fitting for both learned stages. It won all nine comparisons; the descriptive paired 95% match-bootstrap interval for the difference is **[0.3062, 0.3400]**. Mean macroF1 was 0.8961 versus 0.7528. The separate Sample_Game_3, absent from the implemented training corpus, scored 0.6837 versus 0.3598 on clean tracking with frozen nine-game weights.

The [full report](docs/experiments/quality-audit/REPORT.md) includes per-match results, decoder/stack/duration ablations, five sensor scenarios, aggregate records, a figure and split provenance. These are external scores: historical configuration selection reused the nine games, sensor corruption is simulated, and Metrica labels are event-derived proxies. The model also predicts about 24% fewer possession spells than the reference. Hidden competition-match quality remains unknown. The audit changed neither the v1 configuration nor the team CSV.

## New-provider confirmation and v2

We added 20 public SkillCorner matches, fixing ten for development and ten for confirmation before scoring. After comparing nine decoder configurations and two sensor-mixture learned candidates, the selected **v2 changes the four transition costs from 6 to 9**. Its learned weights remain v1 weights trained on the original nine games. The sensor-mixture candidates did not pass the development gates and are not used.

On the ten confirmation games, v2 achieved **spellF1 0.5524 versus v1 0.5451**, winning nine games and losing one; paired descriptive 95% gain interval **[0.0044, 0.0100]**. Under extra label-blind corruption: **0.4719 versus 0.4613**, ten wins. On known matched-sensor games there is a small regression, **0.6432 versus 0.6466**. SkillCorner references are automatic phase proxies, with a different sensor/annotation process from the competition. See the [full new-provider report](docs/experiments/skillcorner/REPORT.md) for all candidates, losses, source hashes and the predeclared adoption rule. This supports a modest improvement on the new provider and does not establish hidden-match quality or qualification probability.

## Review status

The implementation originates from VV's `model-v1` commit `cc44bfe6bceba39e2e67a03fd32e658cca68bc0e`. The reproduction branch adds setup, explicit output paths, controlled worker counts and verification tooling. V1 remains unchanged. The new v2 retains its learned models/features and changes only the externally validated transition costs.

Technical verification completed on 12 November: two clean-source runs regenerated the real-match CSV byte for byte, passed the official validator and matched VV's supplied SHA-256. Those timings and CSV hash describe original v1. The technical document records source revisions, data provenance and limitations; v2 reproduction has its own verification record. Model-selection provenance clarification remains in team review. The external quality audit above provides additional evidence with explicit selection and domain limitations. The repository stays private until the agreed publication window and must be public before submission. The organizer's submission form is a separate release step.

## Data stays local

Store organizer materials under `data/` or outside the checkout. Do not commit match video, tracking, tagging, homography, credentials, or derived artifacts that allow reconstruction of the data. The match assets have separate competition-only terms.

Do not upload organizer materials or extracts to hosted LLMs. Run data processing locally or on an authorized team-controlled machine, without exposing those materials to third-party hosted models.

## Before submitting

- Validate the exact team-named CSV with the official validator.
- Confirm byte-identical regeneration from the documented command.
- Finish the team-named technical document with the method actually used.
- Check that only code and permitted documentation are present in the public repository.
- Submit one team response through the organizer's form after joint review.
