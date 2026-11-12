# Ruses Cules: technical method

**Review draft, 12 November 2026.** This describes the uploaded v1 implementation. A real-match CSV and clean reproduction receipt have not yet been verified. Method-selection provenance requires confirmation from the model author. This document is not ready for submission.

## Approach

We estimate per-frame probabilities from tracking, refine them with temporal context and decode a sequence of possessions and stoppages. Two histogram gradient boosting classifiers estimate dead-ball probability and home-team possession conditional on live play. Training and inference include swapped-team features; we average direct and swapped predictions to enforce team symmetry.

`mtah/features2.py` produces 87 features: visibility, ball gaps, interpolated ball position and movement, distance to the two nearest players of each team, nearby player counts, team shape, speeds, and temporal summaries. Interpolation supplies features alongside explicit gap indicators; it is not an observed ball position. Team attack direction is inferred from average team positions within each half.

The first stage uses `HistGradientBoostingClassifier`: 250 iterations, learning rate 0.08, at most 31 leaves, minimum 200 samples per leaf, L2 regularization 1.0 and 128 bins. We disable early stopping and fix random seeds. We train on every fourth frame and include swapped-team copies.

The second stage consumes the first-stage probability logits, centered temporal means and past/future shifts. Its classifiers use 150 iterations, at most 15 leaves and minimum 400 samples per leaf. Its training inputs are first-stage predictions produced while excluding the predicted external match from first-stage training. We then fit first-stage models on all external matches for competition inference.

The decoder has four internal states: H, A, dead-after-H and dead-after-A. The latter two remember the owner before a stoppage and both produce D. Expanded-state Viterbi imposes a one-second minimum duration before leaving a state; the final segment may be shorter at a half boundary. It aggregates five frames per decoding step. H/A transitions and entering/leaving dead ball have cost 6.0 in the uploaded configuration. Restart snapping is disabled.

## Assets

Competition inference reads only `tactical/BAR-ATH_tactical_atd.xml` and `.txt`. It maps tracking frame IDs 1..199319 onto output frames 0..199318, converts normalized pitch positions to meters and treats track slots as fragments rather than persistent player identities.

The v1 path does not use the provided video, homography, broadcast recording or Smart Tagging. Earlier experimental modules remain in the repository; they are not part of the documented v1 command. We do not claim to have benchmarked every unused asset.

## External training data and labels

We converted 1,288,806 labelled frames across nine external matches before frame subsampling and team-swap augmentation:

| Source | Matches | Label source |
|---|---|---|
| IDSSE / DFL / Sportec Solutions | J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 | Provider `BallStatus` and `BallPossession` fields in tracking exports; dead status maps to D, otherwise the possession team maps to H/A |
| Metrica Sports sample-data | Sample_Game_1 and Sample_Game_2 | Provider event annotations; `tools/ext_metrica.py` derives possession/stoppage intervals from passes, recoveries, carries, shots, ball-out, fouls and restarts |

The original labels come from the named data providers; we did not independently audit their annotation workflows. Our converters derive class labels automatically from those fields. Metrica event-derived labels are a different proxy from corrected tracking ownership and can contain timing and semantic differences.

IDSSE source: [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data), revision `a715a38dfbaf5f58e431727c2b78d174101a703c`; [Bassek et al., Scientific Data, 2025](https://doi.org/10.1038/s41597-025-04505-y), CC BY 4.0. Metrica source: [sample-data](https://github.com/metrica-sports/sample-data), revision `e706dd506b360d69d9d123d5b8026e7294b13996`; we acknowledge Metrica Sports as required by its README. The manifest records each input file's immutable URL and integrity hash.

`mtah/degrade.py` introduces deterministic ball dropouts, position/speed noise and player-track fragmentation into external tracking. The target visibility and slot counts use public organizer descriptions. Other degradation settings are modeling assumptions or unsupervised statistics of the supplied tracking, not hand labels. The uploaded configuration uses degradation mode v2 and seed 0.

## Possession rules

We let the temporal model resolve loose balls and contacts rather than change possession after each nearest-player change. This is an approximation to controlled possession; the model has no referee signal. A dead-ball segment separates two possessions even if the same team restarts. X marks frames outside the published half boundaries. The second-half kickoff frame 113742 comes from the organizer's public starter code, rather than manual inspection; the earlier graded frames of that half receive D.

## Evaluation and limitations

Decoder optimization matched exhaustive enumeration on 60 short synthetic sequences. The local scorer reproduced the organizer's four published baseline results. These checks establish implementation consistency, not competitive accuracy. Final quality is unknown until the organizer evaluates the hidden full-match labels.

Before release, resolve these review items:

- Confirm how the author selected the uploaded method and parameters. The public five-minute window cannot be used for model selection or tuning. No competitive score is asserted here.
- Recheck any claimed independent second-stage holdout score. `tools/exp_v1.py::stack_one` trains stage two from other matches' OOF predictions whose stage-one training can include the held-out match. A fully independent assessment requires nested match splits throughout both stages. First-stage OOF training for the final stack does not by itself provide that assessment.
- Long ball gaps, missing players and domain differences between training and competition tracking can create false stoppages or fragment possessions. Interpolation and temporal windows may also blur transition timing. Offline features use future frames and cannot serve a live causal feed unchanged.
- Exact equality across platforms is not established. Single-thread model fitting, pinned libraries and fixed random seeds are controls; the two-clean-copy check tests the installed environment.

We do not hand-label or hand-correct the supplied match. We keep competition assets, frame-level intermediate arrays, output CSVs and logs local and outside the public repository. We do not send these assets or extracts to hosted LLMs.

## Reproduction and pending results

Follow the installation and data preparation commands in README, then run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python run.py v1 --jobs 2 --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
.venv/bin/python tools/reproduce.py --data-dir /absolute/path/to/data --jobs 2
```

The main command retrains from the pinned external inputs. The reproduction command starts twice from committed source and raw external inputs, checks both output files and records hashes. Fill in the final source revision, measured run times, output SHA-256 and confirmation against the author's CSV after those checks pass. Keep the full receipt local because it includes private input paths and hashes.
