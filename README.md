# Ruses Cules — More Than a Hack 2027

Current adopted candidate: **uniform_temporal_focus**, branch **codex/short-possession-recovery**, 17 November 2026. **Technically READY**: regression tests, research-to-package parity, four clean-source reproductions and the official validator pass. Organizer submission and repository publication remain on user hold. Hidden-match quality and unconditional eligibility are not certified by this status.

| Ten original development games | Previous weighted_short_gap | Current | Wins |
|---|---:|---:|---:|
| Natural, official Spell F1 | 0.680009 | 0.701652 | 10/10 |
| Original blind stress, seed 7 | 0.667271 | 0.686287 | 9/10 |
| Additional stress, seed 23 XOR match ID | 0.667221 | 0.687005 | 8/10 |
| Additional stress, seed 101 XOR match ID | 0.662958 | 0.688400 | 10/10 |

These are repeated-development results using PFF event-derived references, not a new holdout. Forty scenario outputs represent ten matches. Short-spell recall below three seconds rises from 15.05% to 22.66%; subsecond recovery remains weak. Missing-ball false-live fraction increases slightly. No absence-of-overfitting or finalist-probability claim is made.

The historical +0.02 mean-gain rule failed in the original blind scenario. The user explicitly authorized adoption based on the broader evidence and successful technical checks. The original failed reports are preserved; they were not rewritten as passed experiments. The 18 previously opened audit games and target labels were not read or scored for this release. Their old metrics describe older models only.

Read the [current technical document](docs/releases/uniform-temporal-focus/TECHNICAL.md), [adoption receipt](docs/releases/uniform-temporal-focus/release.json), [data coverage review](docs/releases/uniform-temporal-focus/DATA_REVIEW.md), [frozen experiment](docs/experiments/dead-gap-focus/REPORT.md) and [exact manifest](submission_manifest.json).

## Generate and verify

Use Python 3.12, this branch, and the original data folder containing tactical/BAR-ATH_tactical_atd.xml and .txt. Inference uses frozen bundled weights on CPU. Training data, research caches and target labels are unnecessary.

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

Expected CSV SHA-256: `897db60c18fefe362acee239665136d47a8b2c87a54eb8eae67231b0dcdf9e33`. The generator refuses existing output. Local release: `out/final-ready/` (CSV, technical Markdown, reproduction receipt).

169 regression tests pass. All 40 portable outputs exactly match frozen research predictions. Four clean-source runs (macOS ARM twice, offline Linux ARM and offline emulated Linux AMD64) produce the identical 199,319-row CSV and pass the official validator. This is one physical host and inference reproduction, not repeated weight training. Generation source commit: `98e931ccc2eea3e81bcb8aa4b04b90c49bc64e4e`. Later documentation commits preserve the recorded inference/source hashes.

## Data and limits

Training uses the same 65 external games: seven DFL, two Metrica, 20 SkillCorner and 36 PFF. The activity network uses only those 36 PFF training games. Inherited boosted training registries are subsets of the same union. No development/audit match was moved into training. See [source-access qualifications](docs/PFF_SOURCE.md); original provider access is not equivalent to a standard permissive license.

Data-search priority updated on 17 November 2026 at the user's request: elite professional football, preferably La Liga, followed by other major European first divisions/Champions League and senior international tournaments. University SoccerTrack v2 is deprioritized and must not substitute for evidence of performance in elite football. The earlier frozen data review records discovery, not the current acquisition priority. PFF is the 64-match 2022 World Cup; SkillCorner is the 2024/25 Australian A-League (20 matches in the actual manifest); DFL contains two first-division and five second-division German matches from 2022/23. Metrica sample competitions are anonymized, so their level is unverified. Our ten current development matches are all PFF World Cup matches. Existing training sources are not removed without a controlled development experiment. New validation requires independent tracking/reference provenance, synchronization checks and a preregistered whole-match split. Compare tempo, possession durations, ball visibility and tracking errors instead of assuming league names establish input equivalence. No additional eligible independent La Liga tracking-plus-possession cohort has been verified; the current release remains frozen.

## Rollback

Previous source tag: `codex/rollback-weighted-short-gap-20261117`. Old source, weights and research reports remain committed; its exact manifest and documentation are under `releases/weighted-short-gap/`. Local previous CSV: `out/rollback-weighted-short-gap/Ruses Cules.csv`.

```sh
.venv/bin/python tools/generate_recovery_candidate.py \
  --bundle assets/recovery-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'out/rollback-regenerated/Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py \
  --manifest releases/weighted-short-gap/submission_manifest.json \
  --csv 'out/rollback-regenerated/Ruses Cules.csv'
```

Rollback CSV SHA-256: `1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`. Historical `run.py v4` and `generate_mixed_candidate.py` generate still older versions. Public repository access, provider-access qualifications and team eligibility remain outside the technical readiness claim. Nothing has been sent to the organizers.
