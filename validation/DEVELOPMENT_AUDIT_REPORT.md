# Current candidate audit — 16 November 2026

**CONDITIONALLY READY overall.** The selected weighted_short_gap package passes technical checks and the unchanged meaningful-development-gain gates. No known unresolved P0/P1 implementation defect was found in the tested scope. Public source access, original-provider source-access qualifications, unverified team eligibility and new-model generalization remain conditions. Organizer submission and repository publication remain on user hold.

This report supersedes the parent verdict for the current candidate. The historical audit, its repaired defects, old independent-match scores and limitations are preserved in PARENT_FINAL_AUDIT_REPORT.md. Those historical scores must not be reassigned to the modified model.

## Acceptance gates

| Gate | Current result | Evidence and limit |
|---|---|---|
| 1 Rules compliance | CONDITIONAL | No target labels used; training registries exclude development/final/bridge fixtures. PFF original-provider access terms remain qualified; eligibility and organizer repository access unresolved. |
| 2 Data alignment | PASS for used inputs | Inherited audited ATD index/count/mapping/cache contracts, period reset and observation-integrity tests pass. |
| 3 Metric parity | PASS | Official scorer used; canonical/SDK-shadow references agree. Original metric contracts retained; official source pinned at f8f2629ff1e0985f81d7e59d8413678514ee1530. |
| 4 Temporal correctness | PASS for tested contracts |136 tests pass without skips/errors/failures, plus60 brute-force decoder comparisons. New period/team-swap/loss-masking/activity contracts included. Short-possession recall remains low. |
| 5 Submission format | PASS | All four clean-source outputs pass official validator;199,319 exact ordered rows,frame/label schema. Exact artifact/source/model gate also checked. |
| 6 Reproducibility | PASS in tested environments; access conditional | Four byte-identical fresh-source outputs:2macOS ARM,1Linux ARM,1emulated Linux AMD64. Linux network disabled; original two tracking files only. One physical host/operator; training not repeated; repository private. |
| 7 Evidence quality | PASS with stated limits | Development improvement, in-sample provider regression, technical contracts and historical holdouts separated. No hidden score or no-overfit claim. |

## Model quality and selection

| Ten reused PFF development matches | Audited boundary102 | weighted_short_gap | Gain | Wins | Worst game gain |
|---|---:|---:|---:|---:|---:|
| Natural Spell F1 |0.655317|0.680009|+0.024692|9/10|−0.002849|
| Fixed blind sensor stress Spell F1 |0.642574|0.667271|+0.024698|9/10|−0.000622|
| Natural macro F1 |0.885608|0.895472|+0.009864|—|—|
| Blind macro F1 |0.877913|0.892163|+0.014250|—|—|

Provider event-derived references are proxies for organizer possession semantics. All ten matches, both scenarios and both reference reconstructions are retained. No six-bridge or twelve-final candidate scores were produced. The selection rule and gates were frozen before composed results; the selected model was frozen before supplementary checks.22 changed policies were evaluated adaptively; the complete failures remain visible. The score increase is development evidence, not independent confirmation of the new model's generalization.

Natural recall of H/A possessions shorter than3s at IoU0.5 rises65/578→87/578 (11.25%→15.05%). Missing-ball false D on live reference decreases20.99%→19.78%; false live on actual D decreases4.42%→2.86%. Blind equivalents improve22.14%→18.28% and4.21%→3.16%. These frame/recall diagnostics do not replace the official multi-threshold Spell F1.

Six preregistered DFL/Metrica/SkillCorner training matches, two scenarios each, all improve and satisfy provider-regression guards. This is explicitly in-sample evidence and is not the six untouched PFF bridges. Portable inference exactly matches all20 development prediction arrays.

## Implemented changes and rejected alternatives

The same65-game neural parent was continued with training-only short-spell/missing-ball weights for3000 steps. A matched uniform-loss continuation isolates the effect of weighting. A64-feature player-only activity expert trained on36PFF training games contributes25% of the dead-state estimate when the ball is missing; new neural contributes75%. Other probabilities keep50/50 inherited/new neural fusion. Minimum live duration0.4s and H/A transition cost4.5 are adopted only as part of the qualifying combined model. Minimum D1s and other transition costs9 are retained.

Endpoint gap bridging and visible-contact corrections failed and are absent from production. Decoder-only and old-network gap changes failed the meaningful-gain gate and are not separately promoted. Objective, model, source and selection provenance are in docs/experiments/spell-weighted-finetune/. Portable code lives in mtah/recovery_candidate.py and mtah/recovery_features.py; tools/generate_recovery_candidate.py is the current generator. Regressions cover the real packaged model as well as synthetic contracts.

## Exact reproduction and checklist

Tested source commit: dfc9e860bd28810d0aa768a584eb518e96cee0b6. Later documentation/adoption changes do not alter hashed inference sources or models. Four successful generation times:35.51s,35.30s,39.27s,52.11s. CSV SHA-256:

`1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0`

```sh
.venv/bin/python tools/generate_recovery_candidate.py \
  --bundle assets/recovery-candidate/bundle.json \
  --data-dir /absolute/path/to/data \
  --output 'Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py --csv 'Ruses Cules.csv'
```

CSV: ready locally in out/recovery-ready/. Technical document: ready and copied there. Frozen-weight inference: reproduced on all listed environments. Source: branch codex/short-possession-recovery, private; public access is pending. Organizer upload: not performed. Current manifest binds source/model/input/output/document hashes; parent package is preserved. Full technical setup is in TECHNICAL_DOCUMENT.md.

## Confidence and stopping scope

Format confidence is supported by exact contracts and the official validator. Implementation confidence is supported by136 passing tests, real-model invariance and exact20-output parity; this is bounded coverage. Reproducibility confidence is supported across OS/architectures on one host, not independent hardware or repeated training. Rules confidence remains conditional on original provider terms, eligibility and public source access. Generalization evidence for this changed model is limited to reused-development gains and in-sample provider regressions; earlier unseen-match evidence is for its parent.

The requested changes are complete and meet the original adoption gates. Further search on the same ten matches would increase selection bias; no additional change is retained without evidence. A fresh provider-diverse labeled evaluation would strengthen the new model's generalization assessment. Short possessions, sensor jitter, long ball occlusions and cross-provider reference definitions remain risks. No finalist probability is assigned.
