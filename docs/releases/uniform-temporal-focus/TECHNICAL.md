# Ruses Cules — possession segmentation with missing-ball specialists

## Inputs and output

The offline model predicts H/A/D from the organizer tactical ATD XML/TXT. H and A identify home and away possession; D is dead ball, and X marks excluded intervals. The submission contains `frame,label` and 199,319 rows indexed 0–199318. ATD one-based frame IDs map to output index frame_id−1. Published playing periods and kickoff metadata determine coverage and pre-restart D. The generator validates frame counts and team references.

Coordinates are converted to metres. Both ball coordinates must be finite for a detection. Feature interpolation, derivatives, smoothing and neural context reset at period boundaries. Missing-ball information remains explicit. XML track slots supply team membership, not persistent identity features. Video, homography and Smart Tagging are not used: this release uses supplied pitch geometry and frozen external-trained models, without a video detector or an event-derived predictor. Target labels are never inference inputs; this release does not inspect or score them.

## Implemented approach

Observed-ball probabilities retain the previous release exactly: 50% inherited boosted-model probabilities and 50% weighted temporal-network probabilities, for dead/live and conditional home/away independently. The boosted branch contains the original nine-game dead/team model, temporal stacking and a 19-game team model. Geometry and contact features encode distances, local counts, team shape, speed, observation gaps and period-local possession context. Team-swapped inference provides symmetry.

Three frozen specialists change missing-ball predictions only:

- A dead/live temporal specialist continued from the previous network with uniformly sampled training windows.
- A conditional H/A temporal specialist continued with additional windows focused on ownership errors during ball gaps. It supplies the entire missing-ball H/A probability.
- A player-activity temporal network, trained without ball coordinates, gap duration, timestamps, identities or other models' outputs. Missing-ball D probability is 75% dead/live specialist and 25% activity network.

The geometry networks run at 5 Hz with 108 standardized features plus finite masks, width 48, seven residual dilation blocks (1–64) and approximately 101.8 seconds of context. The activity network uses 64 player-only features plus finite masks, width 32 and six blocks (1–32), approximately 50.6 seconds of context. Future context is permitted for offline inference. Geometry specialists use the unchanged training-only normalizer; activity normalization uses its own training periods only. Both geometry specialists were continued for 4,000 steps with seed 20261117; the activity network was trained for 4,000 steps with seed 20261118. Exact plans, metadata and weight hashes are committed.

A duration-constrained Viterbi decoder has H, A and two preceding-team-specific dead states; both dead states emit D. Minimum live duration is 0.4 seconds, minimum D duration 1 second, H/A switch cost 4.5 and other transition costs 9, stride 5. Restart snapping is disabled. Decoder settings are unchanged from the previous release. Because decoding is temporal, changed gap probabilities can also change neighboring observed-ball labels. A loose ball or ball flight is not automatically dead; a touch does not necessarily switch possession. H-D-H contains two possessions. There is no extra live-contested class.

The old boosted activity expert is retained for rollback but does not enter the final missing-ball fusion. The inherited 29-game temporal asset is loaded and checked by shared code; its probabilities are not selected by the final fusion. Rejected research variants remain documented rather than silently replacing this frozen architecture.

## Training data and label provenance

The training union is 65 distinct external matches: seven DFL/IDSSE, two Metrica sample games, 20 SkillCorner matches and 36 PFF World Cup matches. Reference-frame counts before augmentation are respectively 1,002,644, 286,162, 2,955,525 and 5,636,115. The geometry specialists use 139 natural/degraded scenarios from those same matches. The activity model uses 72 natural/blind sequences from the same 36 PFF training games. Inherited nine-game and 19-game boosted registries are subsets of the 65-game union, not additional matches.

DFL reference labels use provider BallStatus/BallPossession (CC BY 4.0). Metrica sample events form an event-derived reference proxy with attribution. SkillCorner automatic H/A phases use complementary within-period D as a proxy (MIT, attribution). PFF possession references use the pinned SDK event-state reconstruction. Predictor geometry uses raw visible detections and excludes event-informed/estimated ball coordinates, reference labels and identifiers. None of these proxy references is claimed to be perfect manually annotated possession ground truth.

No target-match labels or development/audit matches enter model fitting or normalization. Development labels have been reused for model selection; this is disclosed below. Original PFF provider access was obtained, but a standard permissive license for that original release has not been established. The Hugging Face mirror's license is not applied to the original data. See `docs/PFF_SOURCE.md`; source-access qualifications remain relevant. No raw external or organizer data are redistributed.

## Development evidence and limits

Results use the official Spell F1 implementation and the ten original PFF development matches. They are independent of fitting, but repeatedly used for selection. The following are means over matches, comparing the previous weighted_short_gap release with uniform_temporal_focus:

| Input scenario | Previous | Final | Improving matches |
|---|---:|---:|---:|
| Natural visible tracking | 0.680009 | 0.701652 | 10/10 |
| Original label-blind stress, seed 7 | 0.667271 | 0.686287 | 9/10 |
| Additional blind stress, seed 23 XOR match ID | 0.667221 | 0.687005 | 8/10 |
| Additional blind stress, seed 101 XOR match ID | 0.662958 | 0.688400 | 10/10 |

Match IDs: 10510 Croatia–Brazil; 3821 Germany–Japan; 10513 England–France; 10516 Croatia–Morocco; 3859 Cameroon–Brazil; 3827 Brazil–Serbia; 10503 Argentina–Australia; 3858 Serbia–Switzerland; 3838 Croatia–Canada; 3842 Brazil–Switzerland. Four scenarios are still ten independent matches, not forty.

Natural short-spell recall below three seconds increases from 87/578 (15.05%) to 131/578 (22.66%). During missing-ball/reference-live intervals, wrong-team fraction falls from 9.73% to 8.83% and false-D fraction from 19.78% to 18.80%. Missing-ball/reference-dead false-live fraction rises from 2.86% to 3.05%. Subsecond recovery remains weak (0/97 natural for the final candidate). These limitations are retained; the improvement is not universal.

The original +0.02 mean-gain gate failed on seed 7 (+0.019015) and supplementary seed 23 (+0.019784). Other registered regression guards passed. Historical failures are unchanged. The user authorized adoption based on consistent gains and successful technical validation instead of treating that effect-size cutoff as absolute. This is an explicit adoption decision, not a retroactively passed experiment. Canonical and SDK-shadow references are alternate reconstructions, not independent datasets.

The 18 previously opened audit games and target labels were not read or scored in this release. Earlier audit scores belong to older models and are not presented as final-candidate results. Long occlusions, subsecond possessions, proxy annotation semantics and competition/provider shift remain weaknesses. There is no claim that overfitting is absent or that hidden performance or finalist selection is guaranteed.

## Reproduction

Use Python 3.12 and branch `release/short-possession-recovery`. Frozen weights are bundled; inference requires no external training download or research cache.

```sh
git clone --branch release/short-possession-recovery --single-branch https://github.com/8Tigran8/silver-octo-winner.git ruses-cules
cd ruses-cules
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_final_candidate.py \
  --bundle assets/final-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py --csv 'Ruses Cules.csv'
```

The data folder must contain `tactical/BAR-ATH_tactical_atd.xml` and `.txt`. The generator refuses existing output. Expected CSV SHA-256: `897db60c18fefe362acee239665136d47a8b2c87a54eb8eae67231b0dcdf9e33`. Inference source commit: `98e931ccc2eea3e81bcb8aa4b04b90c49bc64e4e`; later documentation commits retain these source/model hashes in `submission_manifest.json`.

Four fresh-source runs (macOS ARM twice, offline Linux ARM and offline emulated Linux AMD64) produced the same CSV bytes and passed the official validator. One physical host was used; weight training was not repeated. The 169-test regression suite passed. Portable/research prediction parity and release receipts are recorded beside this document. Official metric/validator bytes were checked against the current organizer repository.

Local delivery files are in `out/final-ready/`. Organizer submission and repository publication remain on user hold. Technical readiness does not resolve public-source availability, source-access qualifications or team eligibility. The old model, CSV, manifest and source tag `release/rollback-weighted-short-gap-20261117` remain available for rollback. Full experiment evidence: `docs/experiments/dead-gap-focus/`; new data review: `DATA_REVIEW.md`.
