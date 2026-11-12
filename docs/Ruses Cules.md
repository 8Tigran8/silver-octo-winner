# Ruses Cules: technical method

**Technical verification completed, 12 November 2026.** The v1 pipeline regenerates the team CSV from raw inputs and passes the official format checks. Two clean-source runs and the author's independently supplied SHA-256 agree. Competition submission and repository publication remain separate release steps.

## Approach

We use tabular boosting to work with the supplied numerical tracking and missing observations without rebuilding a pixel-level detector or persistent player identities. The temporal stack and state decoder address possession continuity that a nearest-player rule cannot represent. We keep the uploaded v1 features and parameters fixed during reproduction.

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

Decoder optimization matched exhaustive enumeration on 60 short synthetic sequences. The local scorer reproduced the organizer's four published baseline results. On the review machine (Apple Silicon, 16 GB RAM, Python 3.12), both training stages completed on all nine external matches in 578.182 seconds using two worker processes and single-thread fitting. That training preflight read no competition input or public GT. These checks establish implementation consistency, not competitive accuracy. Final quality is unknown until the organizer evaluates the hidden full-match labels.

A fixed-configuration external audit used nested match splits across nine games: first-stage test predictions exclude the outer game, and each inner first-stage fit excludes both the outer game and its inner prediction game. Mean official spellF1 was **0.6466** versus **0.3244** for the organizer baseline, with nine wins in nine matches. The descriptive paired 95% bootstrap interval for the difference was **[0.3062, 0.3400]**; macroF1 was 0.8961 versus 0.7528. Removing the stack gave 0.6401 spellF1; removing Viterbi from the stacked model gave 0.4871. The full model predicted 76% as many possession spells as the reference, so merging separate possessions remains a weakness.

Across the same nine games, spellF1 was 0.5011 versus baseline 0.1874 under stronger corruption, 0.5413 versus 0.2215 with label-blind gaps, and 0.6788 versus 0.3284 on clean inputs. A separate public Sample_Game_3, absent from the implemented training corpus, scored 0.6837 versus 0.3598 with frozen full-nine-game weights and clean tracking; its labels are event-derived proxies. No training or retuning used that game. Full protocols, per-game results and ablations are in the [external audit](experiments/quality-audit/REPORT.md).

Method provenance and limitations:

- The model author declares that v1 model/parameter selection used external matches only; the configuration records selection on 10 November. Our reproduction does not load the competition public GT, choose between variants or retune parameters. The author's explanation of version choice is still being clarified in team review; this declaration is not an independent audit of earlier experiments. No competitive score is asserted here.
- The historical `tools/exp_v1.py::stack_one` evaluation can include the outer test match in first-stage fits for inner predictions; its scores are not valid held-out evidence. The new audit in `tools/quality_audit.py` excludes the outer test match throughout both learned stages. Its nine-game scores remain retrospective because the author had already used these games for configuration selection. Matched v2 corruption conditions on external live/dead labels; clean and label-blind scenarios test other input assumptions. Small samples, simulated corruption and differing annotation processes limit transfer to the competition match.
- Long ball gaps, missing players and domain differences between training and competition tracking can create false stoppages or fragment possessions. Interpolation and temporal windows may also blur transition timing. Offline features use future frames and cannot serve a live causal feed unchanged.
- Exact equality across platforms is not established. Single-thread model fitting, pinned libraries and fixed random seeds are controls; the two-clean-copy check tests the installed environment.

We do not hand-label or hand-correct the supplied match. We keep competition assets, frame-level intermediate arrays, output CSVs and logs local and outside the public repository. We do not send these assets or extracts to hosted LLMs.

## Reproduction and verification

Follow the installation and data preparation commands in README, then run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python run.py v1 --jobs 2 --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
.venv/bin/python tools/reproduce.py --data-dir /absolute/path/to/data --jobs 2
```

The main command retrains from the pinned external inputs. The reproduction command starts twice from committed source and raw external inputs, checks both output files and records hashes.

Verified source: `538f43473421aefe7aa10b457b957179bd0915dd`. Later documentation and external-audit tooling do not alter the tested inference/training code, inference dependencies or original source manifests. Both runs independently converted all nine external matches and retrained both model stages; they shared the installed pinned Python environment and the same source input locations. Input SHA-256 hashes remained unchanged.

| Verification | Result |
|---|---|
| Real-input baseline alignment | Byte-identical to organizer `baseline_smooth50.csv` |
| Clean-source run 1 | Official format PASS; 647.224 s including conversion, training and inference |
| Clean-source run 2 | Official format PASS; 667.945 s including conversion, training and inference |
| Output contract | 199,319 ordered rows, exact `frame,label` header, H/A/D/X labels |
| Comparison with author's file | Both clean outputs match the author's supplied SHA-256 |

Team CSV SHA-256:

```text
4d767cd5d677c0dc3ceee7d716cf00e8f10b48920ae0e35eccf61ac1cf33db6c
```

These checks establish file validity and exact regeneration. They do not measure hidden-match prediction quality. Keep the full receipt local because it includes private input paths and hashes.
