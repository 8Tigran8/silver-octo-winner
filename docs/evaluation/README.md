# Frozen evaluation contract: what counts as a better model

**Version 1.0, 13 November 2026. Applies to future candidates after v4.**
V4 remains the verified release baseline. These new thresholds do not retroactively
replace the rules used to select v4. Machine-readable rules and baseline/split
fingerprints are in [policy.json](policy.json).

## Three separate decisions

| Decision | Evidence | What we may claim |
|---|---|---|
| Development improvement | Complete paired whole-game CV against v4; all quality guards pass | Eligible for independent confirmation |
| Prospective generalization | Exactly one frozen candidate passes an unopened reserve | Confirmed improvement on that reserve, with source/dependence limitations |
| Technical release | Official format validator and byte-identical clean generation; inspectable assets and instructions | Reproducible local submission package |

A format PASS alone says nothing about football accuracy. None of these outcomes
proves zero overfitting or qualification on the hidden organizer match.

## Metric and fixed development benchmark

Use the pinned, unmodified official **spellF1**: match complete H/A possessions,
not individual frames. MacroF1 and class H/A/D F1 protect against collateral damage.
Report possession counts/durations, transition errors, timing and memory for diagnosis.
Do not optimize a different proxy and present it as the ranking metric.

Keep the existing five whole-game folds: nine original games with matched, blind
and stress scenarios, plus twenty SkillCorner games with natural and blind
scenarios: **29 games, 67 scenario cells**. Every augmented copy remains with its
game. Each learned component, stack and normalizer must exclude the complete test
fold; inherited components must also use their appropriate excluded-fold weights.
Predictions must be reference-blind, with period boundaries independent of labels.

Average per match before averaging cohorts. The primary composite gives one quarter
each to original matched, original blind, SkillCorner natural and SkillCorner blind;
original stress is a guard. Extra corruptions are not extra independent matches.
The [v4 control](v4_control.json) records the fixed comparison values.

A future candidate must meet **all** these development rules:

- Composite spellF1 gain at least **+0.02** against v4, with at least **21/29 wins**.
- No cohort mean spellF1 or macroF1 loss greater than **0.01**, including stress.
- No cohort mean H/A/D class F1 loss greater than **0.02**.
- Lower-quartile game score must not fall by more than **0.01**; at most two games
  may lose more than **0.05** spellF1. Publish every loss.
- The descriptive paired whole-game bootstrap gain interval has positive lower
  bound. Resample games within source families, preserving paired scenarios.
  This interval does not remove historical selection or shared-team dependence.

The +0.02 threshold is our engineering choice to require a worthwhile gain, **not
an organizer cutoff or a universal significance threshold**. Absolute 0.65/0.70
scores from different providers are not directly interchangeable.

For stochastic experiments, predeclare seeds 20261113/14/15 and report all three.
Choose the deployable seed or ensemble before opening final data. Publish training
versus whole-game-out results, provider-held-out transfer and sensor-gap results;
a small train/CV gap alone does not establish absence of overfit. Record attempted
candidates and failures so that repeated selection remains visible.

## Unopened reserve

All forty previously screened games are now research/regression data, including
WCSF and Metrica3. They cannot become a fresh final test again.

Preserve the original PFF split: **36 train / 10 dev / 12 final / 6 excluded
knockout bridges**. The eight countries in final groups A/H are absent from PFF
train/dev. Original split and intent remain immutable in the
[legacy record](../experiments/more-data/pff_protocol.json). Policy 1.0 explicitly
supersedes its old v3 comparator, HGB-only candidate and +0.005 gates **before
accessing gated matches**: future confirmation compares against frozen v4 and
requires at least +0.02. V3 may be a secondary historical comparison.

First verify access terms, observed-only tracking conversion, periods and full
H/A/D reference semantics on train/dev. Do not open final features or labels to
debug an adapter. A conditional H/A reference is useful diagnostically but cannot
silently replace the full-task endpoint.

Freeze exactly one candidate, deployable weights, source commit, preprocessing,
scenarios and hashes. Then score it against v4 once on all twelve final games:
mean natural spellF1 gain at least **+0.02**, **9/12 wins**, positive mean gain in
each group A and H, no mean natural macroF1 loss greater than 0.01, and no mean
blind-scenario spellF1 loss greater than 0.01. Report the paired interval as
descriptive: these games share teams and represent one additional provider.

Opening consumes the reserve even after failure. Do not pick a runner-up or tune
on final errors; a changed model requires a new untouched reserve. If access or
valid full-task labels are unavailable, independent confirmation stays **PENDING**
and verified v4 remains the release. Historical model selection is not erased.

## Run the development gate checker

It reads aggregate external-game metrics only, never organizer tracking or GT:

```bash
.venv/bin/python tools/check_evaluation.py \
  --scores /absolute/path/to/candidate_cv.json \
  --candidate candidate_name \
  --provenance /absolute/path/to/fold_training_provenance.json \
  --output reports/evaluations/candidate_name.json
```

Score format is the existing temporal CV schema: plan SHA-256 plus records with
candidate, game, family, fold, scenario, official spellF1, macroF1 and class F1.
Each fold provenance lists train/excluded game IDs, plan SHA-256 and checkpoint
SHA-256. See [existing fold provenance](../experiments/temporal/training_provenance.json).
The checker validates declarations; it cannot prove what an undisclosed artifact
actually trained on. Review all component lineage and reference-blindness tests.

Outputs are `KEEP_V4` or `ELIGIBLE_FOR_CONFIRMATION`; final confirmation is always
`NOT ASSESSED`. Provider diagnostics, seed evidence, frozen final execution and
release verification are separate required evidence. It refuses missing/duplicate
cells, wrong folds, invalid metrics, declared reserved-game training and overwriting
receipts. The unchanged-v4 control correctly fails the improvement gate.

After confirmation, regenerate the exact packaged CSV twice from clean source,
verify model/input hashes and official format, and preserve the prior package.
Publication at the agreed time and the organizer form/receipt remain separate
submission requirements. Competition public/private GT never enters fitting,
threshold choice or model comparison.
