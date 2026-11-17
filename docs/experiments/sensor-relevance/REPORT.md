# Sensor relevance and release choice

We recommend uniform_temporal_focus after reviewing the applicability of the VV stress, at the user's request. This explicitly overrides the previous conservative release choice. The original preregistered gate still FAILS for the new model; its results and selected-old decision are unchanged. This is a post-result engineering choice, not a passed confirmatory experiment.

## Evidence

| Input | Target tactical tracking | Stationary synthetic tracks under VV presets |
|---|---:|---:|
| Completely missing player rows | 0% |40% at15Hz;60% at10Hz |
| Mean player count per row |20.49 |22 only on retained rows |
| Median coordinate/provided speed |1.36/1.41m/s |8.43/12.56m/s at0.5m noise;33.72/50.26m/s at2m |
| Ball visible |59.31% |Cadence thinning plus Poisson bursts |

The stationary probe has true speed zero. Independent Gaussian coordinate errors become very large velocity estimates after differencing. The corruptor also leaves all non-observation rows NaN and preserves existing gaps. Thus VV presets are severe combined failures, not a validated simulation of the supplied tracking. On training PFF3816 alone, their ball visibility falls to12.26%/18.46%, compared with35.04% natural and59.31% in the target. These summaries are not a calibrated estimate of target accuracy or proof of smooth tracks being correct.

The target has genuine long ball gaps: median10.72s,95th percentile50.80s across gaps within published playing blocks. Smooth player tracks do not remove this risk. Model selection rests on the full prior evidence: the new model improves all18 natural games,17/18 original blind-stress games, and short-possession recall14.23%→23.77%. It reduces missing-ball wrong-team fraction11.12%→9.94% and false-dead18.88%→17.70% in natural audit inputs; false-live rises2.61%→2.87%.

Mean SpellF1: original tracking0.639234→0.674644, original blind stress0.619814→0.655080. VV stresses lose0.00593–0.01229 on average. The natural-minus-stress trade-off favors the new model unless one gives extreme synthetic cases a dominant weight; we have no empirical probability estimate for those cases. Minimax performance would prefer the old model, but that is a different objective from expected score on the provided input.

We used automatic target geometry only, no target possession labels. No model scoring, fitting or tuning occurs in audit_sensor_relevance.py; a guard blocks reference paths. The PFF sensor comparison uses training game3816 only. The18 quality results are reused from the prior frozen comparison without recomputation or exclusion. Input summaries and stationary diagnostics do not prove generalization or finalist probability.

Full numerical evidence:results.json; declared diagnostic scope:plan.json; explicit decision and original gate hash:selection_override.json. Old package and manifest remain available for rollback. Nothing has been submitted or made public.
