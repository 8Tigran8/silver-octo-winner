# Confirmed defects and residual risks

| Severity | Finding | Evidence | Action/status |
|---|---|---|---|
| P1 | Base interpolation, derivatives and rolling features were not isolated by period | Synthetic other-half mutation changed 615/26100 first-half feature cells | Fixed: per-period base feature computation. |
| P1 | Second-stage stacking lacked period boundaries | `stack.predict` called on full posterior arrays; its shifts/windows crossed boundaries | Fixed for deployed v4/mixed path by explicit spans; full-predictor isolation test passes. Legacy historical research scripts without span argument retain historical behaviour. |
| P1 | Cached ATD could silently survive changed input bytes with old mtime | Synthetic ball value 0.3 remained after input changed to 0.9 | Fixed: XML+raw SHA-256 cache key. Cold/new-env and warm-cache runs agree. |
| P2 | Equidistant closest-player speed depended on slot ordering | Swap tied speed-1 and speed-7 observations changed 600 feature values | Fixed: mean over tied finite detections. |
| P2 | Short input windows crashed feature stacking | 1/2/12/50-frame synthetic periods | Fixed: rolling kernels capped to input length. |
| P2 | One-second minimum suppresses true very short possessions | Explicit short-possession regression records non-equivalence | Deliberately retained model constraint, not hidden as a passing semantic claim. Changing it is model development, not a bookkeeping repair. |
| P2 | Sensitivity to independent coordinate jitter | Prior whole-fold sensor pilot, six combined cases lose vs v4 | Unresolved quality risk; same target output does not remove it. |
| P2 | Small-source overfit and reference-definition shift | Prior two-game Metrica neural optimism gap ~0.151; PFF event-proxy references | Unresolved; no global no-overfitting claim. |
| P3 | Legacy metric module docstring implied official code is never run | Actual development scripts call official evaluator; synthetic parity now checked | Documentation clarified in audit; helper is not metric authority. |

No confirmed use of private target labels, target training or manual target corrections was found in inspected inference, training registries and code searches. This is a bounded forensic conclusion, not a proof of all historical human actions. A P0 would be declared if such evidence were discovered.

Source/weight freeze from the earlier final remains preserved in Git and the pre-audit archive. Five inference files were repaired; previous source hashes are therefore intentionally superseded for this audit branch. Model weights, decoder parameters and training registry are unchanged. New source identity is pinned in `submission_manifest.json`. Exact target CSV identity is independently checked, not assumed.
