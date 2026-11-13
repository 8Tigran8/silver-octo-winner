# Ruses Cules: v4 technical method

**13 November2026. V4 temporal fusion passed the registered development and external checks; frozen-weight CPU reproduction is being completed.** The final local package and submission status are recorded separately. No hidden competition score is claimed.

## Approach and assets

We infer team possession and stoppages from the supplied automatic tactical EPTS XML/TXT. Input IDs1..199319 map to output frames0..199318. Normalized coordinates become pitch metres; track slots are fragments, not stable player identities. The method uses no video, homography or Smart Tagging and makes no manual correction of this match.

A framewise classifier can lose possession continuity when the ball disappears. We add a learned temporal model and retain the externally validated boosted path through a fixed probability mixture:

```text
Tracking ->87geometry/gap/motion features ->nine-game boosted models+stack ->dead, home|live
          +21observed-contact features ->nineteen-game team head ------->v3 probabilities
          +108missing masks ->25second residual temporal network ------>neural probabilities
                            fixed50/50 probability mixture ->duration Viterbi ->CSV
```

The original path has two histogram gradient boosting classifiers (dead/live and conditional home/away), each250iterations, learning rate0.08,31leaves, minleaf200,L2=1,128bins. Its temporal stack uses150iterations,15leaves,minleaf400 and out-of-match first-stage predictions. Final original weights train on nine public games. The additional nineteen-game team head uses the original250-iteration settings and averages direct/team-swapped predictions. Its conditional probability is blended50/50 with the original stack; it does not replace that path's dead classifier.

The neural model consumes the same108geometry/contact-context features plus explicit108finite-value masks at5Hz. It has48channels and five residual blocks, each two3-tap convolutions with dilations1/2/4/8/16 and dropout0.15. The convolutional receptive field is125steps (25seconds); geometric features also contain noncausal observed-ball context. Two output logits model dead probability and home possession conditional on live play. Direct and team-swapped inference enforce home/away symmetry. V4 averages both neural probability channels50/50 with v3, before decoding.

Training uses fixed-seed20261113,1500AdamW updates, learning rate0.001,weight decay0.001,batch16,256supervised5Hzsteps and64context steps on each side. Families are sampled50/50, then games/scenarios uniformly, with random team swaps. Means/stds fit only training-fold observations; std floors at0.01, standardized values clip to[-8,8], and missing values become zero with a separate mask. Unknown targets contribute no gradient. No validation-based early stopping is used. No match/team/player identity or source event field enters features.

This is our tracking-based temporal implementation, informed by [temporal possession modelling](https://doi.org/10.1016/j.eswa.2023.119780). It is not a reproduction of that paper's confidential-data results. We inspected official PathCRF code/checkpoint and ran a synthetic compatibility check; PathCRF and video-demo weights are not part of v4. See the [research comparison](experiments/temporal/INDUSTRY.md).

## Training data and reference provenance

Every trained component uses public games outside the provided match. The temporal branch uses29matches with4,244,331reference-period frames before5Hzsampling and scenario augmentation. The original and nineteen-game components retain their earlier training registries.

| Source | Amount and annotation | Use and terms |
|---|---|---|
| IDSSE/DFL/Sportec | Seven club games,1,002,644H/A/Dframes; provider tracking BallStatus/BallPossession | Original boosted path and new temporal branch; CC BY4.0 |
| Metrica sample-data | Sample_Game1/2,286,162H/A/Dframes; provider events automatically converted into possession/stoppage intervals | Original boosted path and new temporal branch; event proxies, not corrected-tracking truth; source acknowledgement |
| SkillCorner Open Data | Twenty A-League2024/25games,2,955,525reference-period frames; automatic phases provide H/A, their complement within metadata periods supplies a D proxy | Temporal branch uses all20; v3 team head uses its designated10development games only. Neural dead-proxy loss has weight0.25; original labels and known-live SkillCorner frames weight1. MIT; credit SkillCorner/PySport |

Sources: [IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data), revision`a715a38dfbaf5f58e431727c2b78d174101a703c`, [Bassek et al.](https://doi.org/10.1038/s41597-025-04505-y); [Metrica](https://github.com/metrica-sports/sample-data), revision`e706dd506b360d69d9d123d5b8026e7294b13996`; [SkillCorner](https://github.com/SkillCorner/opendata), revision`4340d274572876239c154c90bc507a9b3250a656`. Manifests pin filenames, sizes and hashes. Provider annotation workflows were not independently audited.

Original games contribute matched-seed0,label-blind-seed1 and clean tracking scenarios. SkillCorner contributes natural detections and label-blind-seed7. Sensor augmentation simulates missing ball observations, noisy positions and fragmented player slots. Matched augmentation uses external labels to approximate sensor dropout; label-blind tests avoid that assumption. SkillCorner resampling preserves missing detections and never uses extrapolated positions as observations. All augmentations of a game remain in its training/test group.

## Possession rules and decoder

The model estimates controlled possession from player/ball geometry and learned context; a nearest touch does not force a hand-over. Four internal decoder states are H,A,dead-after-H,dead-after-A; both dead states outputD. The cost9 duration decoder aggregates fiveframes per step and imposes one-second minimum live/dead intervals, truncated at a period boundary. Restart snapping is disabled. A stoppage separates two possessions even when the same team restarts.

Output is exactly199,319ordered frame,label rows usingH/A/D/X. Outside published halves outputX. The public starter's second-half kickoff frame113742 supplies alignment; earlier graded frames of that half outputD. These boundaries are organizer-published, not hand labels. No competition public/private GT is used to train or select v4.

## Validation and overfit checks

Five whole-game neural folds cover29games, with23or24training games each. Normalization excludes the test fold. Baseline v3 uses its own excluded-game models on the original19 and frozen weights on the ten games outside its training; the original stack retains nested exclusions. Learned v3 probabilities never enter neural training: fusion occurs after prediction. All67baseline scenario scores reproduced earlier values exactly.

| Excluded-game cohort | V3 spellF1 | V4 spellF1 |
|---|---:|---:|
| Original9,matched |0.6414|0.6440|
| Original9,label-blind |0.5419|0.5583|
| Original9,stress |0.5105|0.5320|
| SkillCorner20,natural |0.5817|0.6500|
| SkillCorner20,extra label-blind gaps |0.4978|0.6051|

The predeclared equal-quarter composite of matched/blind original and natural/blind SkillCorner improves0.5657→0.6143; paired descriptive95%gain interval[0.0434,0.0540],26/29game wins. Whole games are resampled within source families, preserving paired scenarios. These are reused research games, so intervals do not account for historical model selection or opponent dependence. DFL matched mean falls slightly,0.6395→0.6379; improvements are not universal.

For the standalone neural branch, full-fit versus excluded-fold spellF1 is0.6292/0.6159 onDFL,0.6498/0.6277 onMetrica and0.6789/0.6662 onSkillCorner. Only two Metrica training games exist. Training on only one provider transfers substantially worse to another: DFL-only→SkillCorner20 scores0.5850; SkillCorner-only→DFL7 scores0.5081. Small train/CV gaps cannot prove absence of overfit; source/sensor/label shift remains material.

The candidate and full checkpoint were frozen before new-source scoring. Additional checks cover eleven games absent from neural training:

| External source | V3→V4 | Scope |
|---|---|---|
| New Groningen/SciSports research match |spellF1:0.5368→0.6858; with extra gaps0.4587→0.5702|One anonymized game, validation only; supplied AliveFlag/PossessionFlag references |
| MetricaSample3 |spellF1:0.5481→0.5527; extra gaps0.3976→0.4614|Previously v1-scored, not fresh; event-derived references, whole-stream prediction without annotation-derived half cuts |
| Nine European SkillCorner games |conditionalH/A macroF1:0.8372→0.8410,8/9wins|Previously used in v3 validation; no D reference, not whole-match spellF1 |

Thus40screened games are evaluated, not40fresh independent tests. [Groningen/SciSports](https://github.com/MatKempeGroningen/WCSF_SolutionRoom) publishes a research sample with attribution requested and no explicit licence file; no raw files are redistributed or used to fit v4. Predictor coordinates/periods are independent of its reference flags. Contradictory source rows at110timestamps are markedX, excluding266/143431period frames; unknown holes affect spell boundaries. Raw-trajectory screens found no suspicious duplicates, but cannot prove anonymous identities. PFF64access is still pending and its future country-separated holdout is unopened.

Full exclusions, per-game scores, source issues, fixed plans and deployment checks are in the [temporal report](experiments/temporal/REPORT.md). Long occlusions, incomplete players, ambiguous control and provider semantics remain failure modes. The method is offline and uses future context. Hidden competition quality and qualification probability are unknown.

## Reproduction

Use Python3.12 and requirements-temporal.txt. The repository includes3.8MBof frozen models trained only on the public external games above. Their hashes and feature-schema hash are checked before loading. PyTorch state uses weights-only deserialization. No private match input or fitted target statistic enters these model artifacts.

```bash
.venv/bin/python run.py v4 --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
.venv/bin/python tools/reproduce.py --data-dir /absolute/path/to/data --method v4
```

Default generation runs onCPU with frozen weights; it does not retrain. Two clean-source CPU generation checks are pending. Research neural training used deterministic MPS operations; optional raw-public-data preparation/fitting is documented in README. Raw-feature equality was checked, and independent neural retraining is being checked separately. Training artifacts need not match across devices; the submitted file is regenerated with the published frozen assets.

Competition materials, predictions and full local receipts remain outsideGit and are never sent to hostedLLMs. Publication and the organizer form are separate release steps; this document does not claim submission.
