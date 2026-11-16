# Continuous validation — 16 November 2026

This work continues after the initial conditional audit; it does not reinterpret green unit tests as high match quality. Production weights, fusion, decoder and the already consumed public/final diagnostics remain frozen.

## Observation correctness

Problem: visibility used only x, so an incomplete XY observation altered visibility/gap features despite lacking a usable location. With exactly one ball observation, interpolation returned zero instead of retaining the observation. These errors can change distances, temporal evidence and therefore possession boundaries.

Hypothesis: require finite XY pairs and preserve a lone endpoint. Before repair, independent synthetic equivalence/interpolation assertions fail. After repair, partial XY and fully missing observations produce identical features and complete-model probabilities; one observation remains at its measured position. Infinite numeric values become missing; malformed player groups raise a line-numbered error. No pitch clamping or target-specific rule is introduced.

Decision: **KEEP as objective correctness repairs**. Ten familiar PFF development games × two observation scenarios × two reference reconstructions retain exactly the preceding Spell F1 and macroF1 values. This is regression evidence, not improved real-match quality or fresh generalization.

## Disposable cache reliability

Problem: truncation, wrong slot dimensions and invalid cached frame IDs can block or misdirect inference. Source-content identity alone does not validate a stored archive.

Hypothesis: validate cache array shapes/types/indices, rebuild invalid archives from original source, and atomically replace completed writes. Three corruption cases fail before repair and recover the original arrays after repair. The key includes a parser-contract version, avoiding reuse of pre-repair parsing. Source parsing errors still fail; they are not swallowed as valid data.

Decision: **KEEP**. Cold and warm full-match generation agree. This is reliability evidence; it is not resistance to a malicious structurally valid cache replacement, nor an improvement in possession quality.

## Whole-model synthetic geometry

Construction was independent of weights: 22 players follow bounded smooth trajectories; the ball is 0.5 metres from a known controller; the other team stays distant. Both H/A constant-control cases are recognized (synthetic Spell F1 1.0). A ten-second missing-ball interval creates false D and reduces the score to 0.4. Partial-Y and fully missing observations now agree exactly, but agreement does not remove the learned false-stoppage weakness.

These are toy observations with constructed truth, not independent football-match validation. They motivate the separately registered controlled-gap study, never direct production tuning. The original sparse four-player all-D fixture remains documented rather than replaced or hidden.

## Controlled-gap experiment

The committed plan `docs/experiments/controlled-gap/plan.json` fixes two maximum gap lengths (2/10 seconds), a common nearest player at both endpoints within 2 metres, finite player path, speed ≤12 m/s, period isolation, and unchanged learned model/decoder. A possession can still stop while the same player remains nearby: the hypothesis is uncertain and must pass actual external development guards.

The required gain is at least +0.02 mean Spell F1, at least 8/10 positive games, worst-game gain ≥−0.02 and mean macro gain ≥−0.002 in both natural and blind scenarios. Final/public/bridge data are prohibited. All 80 records completed: natural Spell changes are −0.000057 / −0.000385, blind changes zero, and neither candidate passes. **REJECT both; production inference unchanged.** See `docs/experiments/controlled-gap/REPORT.md`. A toy-scenario improvement cannot override these measured results.

## Clean environments

Fresh archived source in the separately installed macOS venv generates the same file twice. A separately installed Linux ARM runtime also generates it twice with networking disabled; only the two original tracking files are mounted read-only. Inputs and output hashes agree. All 114 tests pass on macOS and Linux ARM after installing the separate pinned audit dependencies. The initial Linux full-suite missing-gdown failure is retained; gdown is required by mocked optional-download tests, not inference.

Two further emulated Linux x86-64 runs also reproduce the exact CSV. The six-run hash remains `cfbd16ea35075a199d2401c18f162d4c1f953be4049cd776d502ff42ea389c97`. These tests share one physical host and operator. They are not independent hardware validation, a training reproduction or evidence of high hidden score.

## Stopping boundary

**Subsequent continuation:** the user's request for more real matches prompted a prospective audit of all six unopened PFF bridges. They remain excluded from training/selection, but are now consumed evaluation data. Frozen current Spell F1 is 0.613896 versus v4 0.546054, with six gains; all 36 training fixtures score 0.625242. See [the complete audit](../docs/experiments/unseen-bridge-audit/REPORT.md). This corrects the earlier overly broad fresh-data blocker below. The model and submission are unchanged, and the remaining new-provider/reference uncertainty is not removed by this same-provider match holdout.

No known unaddressed P0/P1 implementation defect remains within the audited input contract. The actionable observation/cache defects have been repaired and reproduced; the new quality hypothesis failed. Previously registered boundary-event and duration-weighted emission studies also failed their promotion gates. Repeating threshold searches on these familiar development matches would not supply fresh generalization evidence.

The overall result remains conditional, under stopping condition **B**, not an assertion that no better algorithm exists. A materially different model requires independently assessable data and validated observation/reference semantics before safe adoption. The available final has already been consumed; the current SkillCorner release has no unused matches; SoccerTrack's auxiliary ball carries reference-event timing. A new detector/player-only adapter plus validated reference conversion is substantial unverified research before the deadline. The current pipeline is technically reproducible, while higher hidden-match quality remains unproven. Public release/submission remain on the user's hold; original-provider terms and team eligibility are separate unresolved conditions.
