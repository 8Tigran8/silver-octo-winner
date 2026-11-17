# Ruses Cules: team possession from tactical tracking

Team possession
from tactical tracking

Missing-ball specialists with duration-constrained temporal decoding

| SELECTED MODEL | OUTPUT | AUDIT SPELL F1 |
| --- | --- | --- |
| uniform_temporal_focus | 199,319 frames / H A D X | 0.6746 on 18 PFF games |

## What we built

We estimate which team owns the ball and when play is dead from tactical ATD XML/TXT. We combine geometry-based predictors with temporal specialists for missing-ball intervals, then decode complete possession segments. The method runs offline on CPU and uses future context.

Pipeline: ATD geometry -> observed/missing-ball predictors -> duration decoder -> H/A/D/X CSV.

## Observed ball and missing ball

When the ball is observed, we retain an equal mixture of the inherited boosted predictors and the weighted temporal network for dead/live and conditional H/A probabilities. During gaps, dead/live uses 75% of a temporal specialist and 25% of a player-activity temporal network. A separate ownership specialist supplies conditional H/A.

Geometry networks use 108 standardized features plus finite masks at 5 Hz, width 48 and seven residual dilation blocks: 101.8 seconds of context. Activity uses 64 player-only features plus masks, width 32 and six blocks: about 50.6 seconds. We average direct and team-swapped inference.

## Input contract

We validate XML teams, track slots and contiguous ATD frame IDs, convert coordinates to metres and reset interpolation, derivatives and neural context at each period. A detection needs two finite coordinates. Missingness remains explicit. Video, homography and Smart Tagging do not enter inference. We exclude target possession labels and human-produced events from fitting and selection.

H = home possession; A = away possession; D = dead ball. X marks excluded intervals from published period metadata. ATD one-based IDs map to zero-based output rows. The CSV header is frame,label.

## Training data and possession rules

We fit on 65 distinct external matches. The inherited nine-game and 19-game boosted registries are overlapping subsets of this union, not additional matches. Development and audit matches remain outside fitting and normalization.

| SOURCE | GAMES | REFERENCE FRAMES | REFERENCE |
| --- | --- | --- | --- |
| IDSSE / DFL | 7 | 1,002,644 | Provider BallStatus / BallPossession |
| Metrica samples 1 and 2 | 2 | 286,162 | Intervals from sample events |
| SkillCorner A-League | 20 | 2,955,525 | Automatic H/A phases; complementary D proxy |
| PFF World Cup 2022 | 36 | 5,636,115 | Pinned SDK event-state reconstruction |

Frame counts precede augmentation. Geometry specialists use 139 natural/degraded scenarios from the same 65 games; the activity network uses 72 natural/blind sequences from the 36 PFF training games. For PFF predictor geometry we use raw visible detections, excluding event-informed/estimated ball positions, labels and identifiers. References have different annotation semantics and include proxies.

## Frozen training configuration

Both geometry specialists continue the parent network for 4,000 steps with seed 20261117 and its training-only normalizer. Dead/live uses uniform training windows; H/A training focuses additional windows on ownership errors during gaps. The activity network trains for 4,000 steps with seed 20261118 and training-period normalization. Its inputs exclude ball coordinates, gap duration, identities, timestamps and other model outputs. Metadata and weight hashes specify the exact configurations.

## Possession and restart decisions

The decoder has H, A, dead-after-H and dead-after-A states; both dead states emit D. Minimum live duration is 0.4 s, minimum D duration 1 s, H/A transition cost 4.5 and other transition costs 9, with stride 5. Restart snapping is disabled. Published period and kickoff metadata determine excluded blocks and pre-restart D.

A loose ball or ball flight does not automatically stop play. A touch need not change the owning team. H-D-H contains two possessions; contested live possession has no separate output class. Temporal decoding means changed gap probabilities can also affect neighboring observed-ball labels.

## Source access and attribution

DFL uses CC BY 4.0; SkillCorner uses MIT with attribution. Metrica supplies sample tracking/events. We obtained original PFF access through its provider; a standard permissive license for that release has not been established, and the mirror's GPL label is not applied to it. Source-access qualifications remain in docs/PFF_SOURCE.md. We exclude raw organizer and external data from the repository.

## Measured performance

We score maximal H/A possession spells with the official metric, averaging F1 over temporal IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7, then average across matches. Frame-level macro F1 and per-class F1 provide additional regression checks.

| 10 DEVELOPMENT GAMES | PREVIOUS | SELECTED | WINS |
| --- | --- | --- | --- |
| Original tracking | 0.6800 | 0.7017 | 10 / 10 |
| Original blind stress, seed 7 | 0.6673 | 0.6863 | 9 / 10 |
| Blind stress, seed 23 XOR match ID | 0.6672 | 0.6870 | 8 / 10 |
| Blind stress, seed 101 XOR match ID | 0.6630 | 0.6884 | 10 / 10 |

These ten matches were repeatedly used for development. Four scenarios still represent ten matches. The original +0.02 effect-size gate failed in two stress cases (+0.0190 and +0.0198); those failures remain in the experiment records.

| 18 PREVIOUSLY OPENED AUDIT GAMES | PREVIOUS | SELECTED | GAIN |
| --- | --- | --- | --- |
| Original tracking | 0.6392 | 0.6746 | +0.0354 |
| Original blind stress | 0.6198 | 0.6551 | +0.0353 |
| VV: 10 Hz / 0.5 m jitter | 0.4854 | 0.4795 | -0.0059 |
| VV: 15 Hz / 0.5 m jitter | 0.4902 | 0.4845 | -0.0057 |
| VV: 10 Hz / 2 m jitter | 0.3995 | 0.3900 | -0.0095 |
| VV: 15 Hz / 2 m jitter | 0.4096 | 0.3974 | -0.0123 |

The selected model wins 18/18 original-tracking games and 17/18 original-blind games. The descriptive paired whole-match bootstrap 95% interval for the original-tracking gain is +0.0270 to +0.0441 (10,000 resamples). Reuse, common teams and one provider limit that interval's interpretation.

## Short spells and ball gaps

| 18 GAMES, ORIGINAL TRACKING | PREVIOUS | SELECTED |
| --- | --- | --- |
| Recall of possessions shorter than 3 s, IoU 0.5 | 14.23% | 23.77% |
| Wrong team / missing-ball reference-live frames | 11.12% | 9.94% |
| False D / missing-ball reference-live frames | 18.88% | 17.70% |
| False LIVE / missing-ball reference-dead frames | 2.61% | 2.87% |

The 18 games did not enter fitting, but were previously opened and now inform release selection. They are not a fresh prospective holdout. Canonical and SDK-shadow references reconstruct the same events. Full match IDs, failures and diagnostics: docs/experiments/final-opened-comparison/REPORT.md.

## Why we selected this version

We first rolled back under a conservative rule requiring the new model to avoid regressions on all four VV stress presets. After a sensor-relevance review, we explicitly overrode that release choice. The original gate still FAILS for this model; we did not rewrite it as a passed experiment.

## What the additional stress changes

Each preset combines 10/15 Hz observation thinning, independent Gaussian x/y jitter of 0.5/2 m for players and ball, and two Poisson ball-gap starts per minute with 2-6 s gaps. It preserves existing gaps and leaves non-observation grid rows empty, without holding or interpolation. The presets have not been calibrated to the organizer's tracking error distribution.

| INPUT DIAGNOSTIC | SUPPLIED TRACKING | STATIONARY VV PROBE |
| --- | --- | --- |
| Rows with all players missing | 0% | 40-60% |
| Median player speed | 1.36 m/s from coordinates | 8.43 / 12.56 m/s at 0.5 m jitter |
| Median apparent speed at 2 m jitter | Not applied to target | 33.72 / 50.26 m/s |
| Mean players per supplied row | 20.49 | 22 only on retained rows |

These diagnostics use automatic tactical geometry and synthetic stationary tracks only, with no target possession labels. They show that the presets create severe combined corruption unlike the supplied smooth player input. Smoothness is not proof of correct positions or possession labels. No model weights, decoder parameters or predictions changed during the review.

## The trade-off

We favor the consistent original/original-blind gains and improved short-spell/gap decisions over a veto based on these uncalibrated failures. The older model remains preferable under a minimax objective over the four extreme presets. The selected model has stronger measured results on ordinary PFF tracking. We have no empirical probability model that proves which one maximizes hidden-match score.

This is a post-result engineering choice and introduces additional selection bias. It is documented in docs/experiments/sensor-relevance/selection_override.json. The prior comparison results and rollback package remain available.

## Limits that still matter

The target has long real ball gaps: median 10.72 s and 95th percentile 50.80 s within the published playing blocks. Most short possessions remain unresolved; the selected model matched 0/97 subsecond spells on the ten natural development games. Independent La Liga evaluation is absent, source labels are proxies, and all validation quality comes from one provider/tournament. We cannot infer hidden accuracy or a finalist probability.

## Reproduction and handoff

Use Python 3.12 and the main branch. The frozen CPU weights require only tactical/BAR-ATH_tactical_atd.xml and .txt from the organizer data bundle. Training downloads and research caches are unnecessary.

```sh
git clone --branch main --single-branch \
  https://github.com/8Tigran8/silver-octo-winner.git ruses-cules
cd ruses-cules
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_final_candidate.py \
  --bundle assets/final-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py \
  --csv 'Ruses Cules.csv'
```

## Exact output

The generator refuses an existing output. The CSV has frame,label and 199,319 contiguous rows. Expected SHA-256:

897db60c18fefe362acee239665136d47a8b2c87a54eb8eae67231b0dcdf9e33

## Verification evidence

Four clean-source CPU generations (macOS ARM twice, offline Linux ARM and emulated Linux AMD64) produced identical bytes and passed the official validator. These runs used one physical host. Our teammate separately reported two matching runs for this candidate on a Linux ARM server, with the same hash and validator PASS; that external evidence is a teammate report. The 176-test regression suite passes. These checks reproduce inference, not weight training or hidden-match accuracy.

The manifest binds inference source, model weights, organizer inputs and the output hash. Generation source: 98e931ccc2eea3e81bcb8aa4b04b90c49bc64e4e. Later documentation and decision revisions retain the same inference/model bytes. Old weighted_short_gap remains available under releases/weighted-short-gap/.

## Sources and full records

DFL: huggingface.co/datasets/pysport/idsse-data (Bassek et al., 2025). Metrica: github.com/metrica-sports/sample-data. SkillCorner: github.com/SkillCorner/opendata. PFF: blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset. Exact source files, versions, SDK pins and training registries are recorded in the repository; source-access qualifications are in docs/PFF_SOURCE.md.

Official protocol: github.com/metrica-sports/more-than-a-hack-2027. Frozen comparison: docs/experiments/final-opened-comparison/. Sensor review and explicit choice: docs/experiments/sensor-relevance/. Portable/research parity and clean-source receipts: docs/releases/uniform-temporal-focus/.
