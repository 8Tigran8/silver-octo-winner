# Ruses Cules: technical method

**V3 verified candidate, 13 November 2026. External validation and two clean-source reproductions passed.** V3 keeps the original dead-ball model and adds a team-only head trained on nineteen public matches. Repository publication and competition submission are separate release steps.

## Method and inputs

The command reads only the competition tactical EPTS XML/TXT. It maps input frame IDs 1..199319 to output frames 0..199318 and converts normalized positions to metres. Track slots are fragments, not persistent player identities. V3 does not use the supplied video, broadcast, homography or Smart Tagging. No hand correction or annotation of the provided match enters the method.

We choose tabular boosting because the input already supplies numerical tracking, with missing ball/player observations and fragmented identities. The pipeline has two probability paths followed by a temporal decoder:

```text
Tracking -> 87 geometry/gap/motion features -> nine-game boosting + temporal stack -> qD, qH
             + 21 observed-contact features -> nineteen-game team-only boosting -> rH
                                                qD, 0.5*qH + 0.5*rH -> Viterbi -> CSV
```

The original path uses two `HistGradientBoostingClassifier` models: dead-ball probability and home possession conditional on live play. Each uses 250 iterations, learning rate 0.08, at most 31 leaves, minimum 200 samples per leaf, L2=1 and 128 bins. The temporal stack consumes first-stage logits, centered means and past/future shifts; its classifiers use 150 iterations, at most 15 leaves and minimum 400 samples per leaf. First-stage out-of-match predictions train this stack; final first-stage weights fit all nine original games. Seeds are fixed at zero, early stopping disabled, fitting single-threaded.

`mtah/features2.py` supplies 87 features: visibility/gap durations, ball position and speed, nearest-player distances/counts by team, team shape/activity and temporal summaries. Interpolation supplies features alongside gap indicators, not observations. Team attack direction is inferred from mean team positions within each half.

`mtah/possession_context.py` adds 21 features to the new team-only classifier: previous/following visible-frame team-distance margins, nearest distances/counts and estimated ball speeds; decayed endpoint evidence; endpoint agreement/change; visible-only rolling margins. These carries/windows reset at period boundaries. This classifier uses the original 250-iteration settings. Training and inference include team swaps; averaging direct and swapped predictions enforces home/away symmetry. We retain the original stacked qD and blend the two conditional home probabilities with fixed equal weights. No new dead-ball head is trained from SkillCorner labels.

The duration-constrained Viterbi decoder has H, A, dead-after-H and dead-after-A internal states; both dead states output D. It aggregates five frames per step, uses transition costs 9 and one-second minimum live/dead durations. A final segment may truncate at a period boundary. Restart snapping is disabled. Original v1/v2 commands remain available.

## Training data and label provenance

The original path trains on nine matches, with 1,288,806 H/A/D-labelled frames before sampling. The added head uses 1,495,434 unique live-team frames across nineteen matches, producing 747,644 sampled examples after sensor augmentation and team swaps.

| Public source | Games and amount | Labels and use |
|---|---|---|
| IDSSE / DFL / Sportec | Seven: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 | Provider tracking `BallStatus` / `BallPossession`; original path uses H/A/D, new head uses H/A only |
| Metrica Sports sample-data | Sample_Game_1 and Sample_Game_2; together with DFL, 735,744 live-team frames | Provider events automatically converted into possession/stoppage intervals; event-derived proxies differ from corrected tracking |
| SkillCorner Open Data | Ten 2024/25 development matches listed in `mtah/expanded_team_training.json`; 759,690 live-team frames | Provider automatic possession-phase intervals supply H/A; these train only the new team head |

Original input sampling uses every fourth frame. The new head uses matched-seed0 / label-blind-seed1 / clean versions of each original game, stride 12 each, and natural / label-blind-seed7 versions of each SkillCorner game, stride 8 each. `mtah/degrade.py` simulates ball gaps, position/speed noise and player fragmentation. Matched v2 simulation uses external live/dead labels to model sensor dropout; label-blind simulation chooses gaps from geometry. These are modelling assumptions, not evidence of the target's true label distribution.

IDSSE: [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data), revision `a715a38dfbaf5f58e431727c2b78d174101a703c`, [Bassek et al. (2025)](https://doi.org/10.1038/s41597-025-04505-y), CC BY 4.0. Metrica: [sample-data](https://github.com/metrica-sports/sample-data), revision `e706dd506b360d69d9d123d5b8026e7294b13996`; we acknowledge Metrica Sports. SkillCorner: [Open Data](https://github.com/SkillCorner/opendata), revision `4340d274572876239c154c90bc507a9b3250a656`, MIT; credit SkillCorner/PySport. The source manifests pin URLs, sizes and hashes. Provider annotation workflows were not independently audited.

SkillCorner input uses only detected positions. Adjacent finite observations resample from 10 to 25 Hz, preserving missing data. Automatic phase intervals intersect their declared metadata period. Their complement is a D proxy for evaluation, not a target copied into predictions. Neither phase labels nor tracking possession fields enter model features. The other ten modern SkillCorner games and nine European 2019/20 games are validation-only; the latter have H/A references only where automatic ownership is known, with unknown ownership X and no invented D labels.

## Possession rules and evaluation

The model approximates controlled possession from geometry and temporal evidence; nearest contact alone does not force a hand-over. A D interval separates possessions even if the same team restarts. Outside published halves we output X. The public starter's second-half kickoff frame 113742 supplies the boundary; earlier graded frames of that half receive D. This is organizer-provided alignment, not a hand-labelled rule.

We fit twenty new heads: one on all nineteen games and nineteen leave-one-whole-game-out heads, excluding every augmentation of the scored game. Original-game baseline stack probabilities use the previous nested exclusions. We select among four fixed blends using development data, freeze one candidate, then evaluate it once on the nine European games. The protocol, exclusions, hashes and per-game results are in the [expanded-training report](experiments/expanded-team/REPORT.md).

| Validation cohort | Metric | V2 | V3 | Scope |
|---|---|---:|---:|---|
| Nine new European games | Conditional H/A macroF1 | 0.8203 | 0.8372 | 9/9 wins; paired descriptive 95% gain interval [0.0134, 0.0209]; no D reference |
| Prior ten SkillCorner confirmation games | Whole-spell F1 | 0.5524 | 0.6046 | 10/10 wins; gain interval [0.0408, 0.0628]; reused secondary cohort |
| Same ten with extra label-blind corruption | Whole-spell F1 | 0.4719 | 0.5156 | 10/10 wins; reused secondary cohort |

The selected blend passed the predeclared development and confirmation gates. Known matched-sensor spellF1 decreases slightly, 0.6432 to 0.6414. The previous seventeen context/augmentation policies did not clear their selection gate and are not the submitted method. Earlier v1/v2 results and negative experiments remain documented in the linked reports.

These sources have different sensors and label semantics. Development reused known games and does not independently validate the complete historical hyperparameter-selection process. The European endpoint tests known-live H/A predictions, not full-match spellF1 or dead-ball accuracy. The secondary ten games were previously used to confirm v2; they were not training inputs for either method, but are not pristine new data. Intervals resample whole games, 10,000 draws, seed 20261113, and are descriptive. No hidden-match quality or qualification probability is claimed.

The original model author declares external-only parameter selection; this is not an independent audit of all historical version choices. The historical non-nested stack evaluation is blocked by default; current estimator exclusions are recorded. No competition public GT is used to fit or select v3. Long occlusions, ambiguous contacts and domain differences remain failure modes. Offline features use future frames and require redesign for live causal use.

## Reproduction and technical checks

Use Python 3.12 and the pinned requirements from README, fetch/convert the original nine public games, fetch the ten extra training games with `tools/fetch_skillcorner.py --only-training`, and obtain the private tactical XML/TXT from the organizer.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python run.py v3 --jobs 2 --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
.venv/bin/python tools/reproduce.py --data-dir /absolute/path/to/data --method v3 --jobs 2
```

The command retrains both original stages and the new nineteen-game head, then writes exactly 199,319 ordered `frame,label` rows. The verifier exports committed source twice, independently converts raw inputs, retrains, validates and compares CSV bytes while checking input hashes. It never scores the competition public window. Matching bytes on the installed pinned environment do not establish equality across platforms.

V3 preflight and both clean-source runs passed the official format check. Setting the added-head weight to zero reproduced the verified v2 CSV byte for byte. Tested source: `6f067004dc59c45ea7e21be6a7bd4246c7e20651`. Both independently retrained runs produced SHA-256 `c501145adc08873ae0bab30e88e9cd2cfa88484210c26bed1e6f9fa667724e40`, matching preflight. Timings: 513.040 s and 511.716 s, four workers on Apple M4 / 16 GB. All raw input hashes stayed unchanged. The [v3 receipt](experiments/expanded-team/v3_reproduction.json) records the result; the previous [v2 receipt](experiments/skillcorner/v2_reproduction.json) is preserved.

Competition materials, intermediates, outputs and full receipts stay local and outside the public repository. We do not send match materials or extracts to hosted LLMs. Submission and repository publication require their separate release steps.
