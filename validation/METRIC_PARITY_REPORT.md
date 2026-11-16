# Exact official metric audit

The current official metric SHA-256 is `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`. It exactly matches the pinned evaluator used by external studies. Official executable code is the authority. The legacy `mtah.metrics` helper is checked against it, not substituted for official evaluation.

`tests/test_audit_contracts.py::MetricParity` compares named synthetic cases and 100 fixed random arrays for spellF1, macroF1 and predicted spell counts, to 14 decimal places. Actual results are in `metric_cases.json`. Cases include perfect labels; shifts 1/3/8/25; split/merge; wrong team; short missed/spurious possession; wrong D; H-D-H and H-A-H; internal/transition/leading X; all-X predictions; constant labels; no live possessions; and halftime. Independent semantic assertions verify X filling, D splitting and invariance to ungraded predictions.

The implementation extracts maximal H/A spells inside valid GT runs; fills prediction X forward and then backward at the run head; leaves an entirely abstained run unfilled; greedily matches same-team overlaps one-to-one in descending IoU, breaking ties by indices; averages F1 over 0.3/0.4/0.5/0.6/0.7. MacroF1 averages H/A/D class F1 on GT-known frames and charges prediction X as a miss. It does not average spell F1 by class.

One-frame shifts are not universally free and constant predictions are not universally zero on arbitrary synthetic arrays. Published example numbers describe the organizer dataset; tests use the actual scorer instead of asserting those claims globally.

CSV mutation tests cover index shifts, duplicates, blank labels, unknown labels, float frame dtype and missing rows. A specification nuance: the current official validator accepts additional columns, while our stricter local validator rejects them. Our output has exactly two columns and the exact header, so this disagreement does not invalidate it.

Public diagnostic: exactly one source/output freeze preceded one call to official `run_evaluation.py`. See `public_diagnostic_freeze.json`. **DIAGNOSTIC ONLY — NOT A GENERALIZATION ESTIMATE.** No model comparison or parameter change follows that result.
