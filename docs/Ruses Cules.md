# Ruses Cules: boundary102 technical method

Validated locally on 15 November 2026. The model and local review package are complete; publication and organizer submission remain on user hold. No competition-match quality score is claimed.

## Method and assets

Assign H/A/D/X to exactly199,319ordered frames from the supplied automatic tactical XML/TXT. IDs1..199319map to0..199318. Video, homography and Smart Tagging are unused. The output is algorithmic; the provided game's human annotations, public/private GT and manual corrections are never used for fitting, selection or calibration.

```text
Automatic tracking ->87geometry +21contact features
                   ->frozen9-game boosted path +19-game team head ->v3 probabilities
                   ->108features +108missing masks ->102s TCN ->neural probabilities
                   ->fixed50/50mixture ->cost9duration Viterbi ->H/A/D/X CSV
```

The TCN has48channels and seven residual blocks with two3-tap convolutions per block and dilations1/2/4/8/16/32/64:509steps at5Hz,101.8seconds of receptive field. Dropout is.15. Two logits predict dead and conditional-home probabilities. Direct and team-swapped inference impose H/A symmetry. Both probability channels are mixed50/50with the inherited v3 path. The original v4 neural branch is only a comparison control.

The decoder retains cost9transitions and one-second minimum live/dead durations. Internal states are H,A,dead-after-H,dead-after-A; both dead states output D. Loose balls and challenges are resolved by learned contact/context probabilities and the transition penalty, rather than transferring possession on every nearest touch. A stoppage separates same-team possessions. Restart snapping is disabled. Prediction is offline and uses future context. Published half/kickoff alignment is retained.

## Training and source provenance

The temporal branch trains on65public matches, all outside the provided game. Means/scales fit training observations only; missing masks are explicit, std floors at.01and standardized values clip to[-8,8]. Training uses seed20261113,6000AdamWupdates, LR.001, decay.001, batch16,256supervised5Hzsteps and256context steps on either side. Four provider groups are sampled equally, then games/scenarios uniformly. Team swaps are random. A fixed.15boundary BCE with positiveweight8uses adjacent H/A/D probability changes; unknown/context pairs are masked. SkillCorner Dproxy loss weight is.25.

| Source | Games / unaugmented H/A/D frames | Labels and terms |
|---|---|---|
| IDSSE / DFL |7 /1,002,644 |Provider BallStatus/BallPossession; CC BY4.0 |
| Metrica samples1/2 |2 /286,162 |Provider events automatically converted to intervals; source acknowledgement |
| SkillCorner A-League |20 /2,955,525 |Automatic phases provide H/A; metadata-period complement is a Dproxy; MIT |
| Original PFF World Cup |36 /5,636,115 |Embedded provider events, separately reconstructed using the primary SDK semantics; original-release access, no raw redistribution |

Total:9,880,446unaugmented reference-period frames. Frames and augmented scenarios are not independent matches. Annotation workflows were not independently audited. The inherited boosted and team models retain their separate9/19-game training registries.

PFF uses raw VISIBLE observations only; ESTIMATED and event-informed smoothed coordinates are excluded. No source event/possession field or identity enters features. References are separate: OUT/ENDmarks dead, kickoff/OTBmarks live, home_ball supplies ownership. Period state resets and duplicate-conflict masking are fixed. The alternate SDK last-row shadow is a sensitivity check, not a score-selected label version. These are event proxies, not corrected frame-by-frame competition truth.

Sources: [IDSSE/DFL paper](https://doi.org/10.1038/s41597-025-04505-y), [Metrica sample-data](https://github.com/metrica-sports/sample-data), [SkillCorner Open Data](https://github.com/SkillCorner/opendata), [PFF original release](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset). Files, sizes, revisions, loader pin and hashes are in the [experiment record](experiments/pff-mixed/REPORT.md) and source manifests. Raw data are not redistributed. Validation-only labelled sources also include Metrica3, WCSF1 and nine European SkillCorner games; the latter have conditional H/A references only.

## Fixed evaluation and results

All learned direct/nested components and normalizers exclude the complete scored fold on the reused29-game/67-cell grid. Actual normalizers and inherited checkpoint lineage are verified. The original PFF split remains36training/10development/12final/6excluded country bridges. Both national teams from final groupsA/Hare absent from PFFtraining/development. One model, code, weights, reader and scenarios were sealed before opening final contents, and remained unchanged throughout the one-shot test.

| Comparison | V4 | boundary102 | Gain | Game wins |
|---|---:|---:|---:|---:|
|29reused development, immutable historical control |.61435 |.64317 |+.02882 |27/29 |
|29same complete-fold exclusions |.61218 |.64317 |+.03099 |29/29 |
|10PFFdevelopment, natural |.61404 |.65523 |+.04119 |10/10 |
|12previously unopened final, natural |.54180 |.60428 |+.06248 |12/12 |
|12final with label-blind sensor stress |.52648 |.58696 |+.06048 |12/12 |

All registered source/cohort, macro, class, stress, lower-tail, large-loss and final gates pass. Final macroF1gain is+.02572; H/A/D F1gains are+.02577/+.03175/+.01964. Group gains areA+.07076andH+.05420. The descriptive paired whole-game95%gain interval is[.04827,.08266]; shared teams and two groups limit independence assumptions. Absolute values across cohorts are not comparable train/test gaps; use paired controls on the same games.

The data-expanded25-second control also passes development, but the combined longer-context/boundary model wins the predeclared selection rule. These two changes are not separately ablated. All11reused external regression guards pass; notable losses remain on noisy Metrica3 spellF1(-.00900)and European conditional H/A macroF1(-.00365), within the fixed.01guard. All prior failed studies are retained.

## Overfit and limits

The standalone neural branch's full-fit/whole-fold-out spellF1 is DFL.68989/.66764, Metrica.85653/.70556and SkillCorner.70864/.71486. The.15097gap on only two Metrica games is a material source-specific overfit risk. This is a descriptive post-selection diagnostic, not the deployed50/50fusion and not a reason to change the frozen model after final opening. The independent final gain supports generalization; it does not prove absence of overfit everywhere.

Short possessions remain under-segmented: PFFdevelopment predicted/reference spell-count ratio is.755; predicted mean duration17.54s versus reference13.75s. Long ball gaps, incomplete players, ambiguous control, provider annotation differences and anonymous WCSF provenance remain limitations. No hidden competition quality or qualification probability is known.

With more time: improve coordinate-jitter robustness using development-only augmentation, collect independent Metrica-like games, reduce the tiny source's sampling weight, and improve short-spell duration modelling. Any changed model requires a new untouched reserve; the consumed final cannot become a tuning set.

## Supplemental sensor diagnostic

A later [two-game sensor pilot](experiments/provider-stress/REPORT.md) loses to same-exclusion v4 in all six distinct combined perturbation cases: spellF1 gains range from -0.00638 to -0.06944. Clean/native controls improve. Post-hoc one-factor probes retain gains for cadence-only/burst-only changes but lose under independent per-observation coordinate noise. This is a material jitter-robustness limitation, not six new independent matches or a new promotion gate. All direct/nested components and neural normalizers exclude the full original fold. The original PFF final, model, mixture, decoder and candidate CSV remain unchanged.

## Exact-file reproduction

Use Python3.12and requirements-temporal.txt. No external training cache/download or retraining is required to generate labels; the repository contains the frozen public-trained assets. Model, metadata and inherited manifest hashes are checked; the new checkpoint uses weights-only deserialization.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_mixed_candidate.py \
  --bundle assets/pff-mixed-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'Ruses Cules.csv'
```

Two clean source generations passed the official validator and returned identical bytes in26.779sand26.838sonCPU, with peak child RSS1.50GiB. This is frozen-weight inference reproduction, not repeated training. Input/model hashes remained unchanged. Tested source:d10dd97438c3; all67frozen files remained unchanged through final validation. The deployment loader also exactly matched the previously scored PFF3821predictions under both sensor cases.

CSV SHA-256: `cfbd16ea35075a199d2401c18f162d4c1f953be4049cd776d502ff42ea389c97`.

The four-page PDF and exact CSV are prepared locally under `out/review-ready/`. Publication and the organizer form have not been performed. Historical v4 remains available under its explicit command and [archived method](experiments/temporal/TECHNICAL_V4.md).
