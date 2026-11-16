# Current team-review status

16 November 2026. Candidate: **boundary102**. Code, frozen weights and current documentation are on `codex/reliability-fusion`. Remote `main` remains the historical v4. The repository remains private; publication and organizer submission remain on user hold.

## Completed

- The original frozen PFF final passes: natural spellF1 0.54180 to 0.60428, 12/12 game wins; fixed blind stress gain +0.06048, 12/12 wins.
- All 67 frozen source/model/evidence files remain unchanged. No weights, mixture or decoder changes follow the final.
- A fresh clone of review commit `072feab1f0db535141f98023ebc59bdaa761cb36` regenerates byte-identical candidate CSVs twice and passes the official format validator. Runs: 28.222 s and 25.946 s on Apple M4 CPU. See [the receipt](experiments/pff-mixed/review_branch_reproduction.json).
- The reproduction wrapper now creates its ignored work directory in a fresh checkout. This is a packaging fix, with no inference changes.
- Forty-one converter/core isolation/reader/gate tests pass. The sensor pilot retains all losses and duplicate inputs; the standalone neural Metrica overfit risk remains documented.
- Current README and technical Markdown are rewritten with readable spacing. PFF original access, its request route, annotation proxies and the unused HF mirror are distinguished in [PFF_SOURCE.md](PFF_SOURCE.md).
- The four-page PDF includes source-access terms, the actual model, results, overfit and jitter weaknesses, and the exact reproduction command. Organizer assets and raw external data are excluded from Git.

## Independent Linux check

VV reported that historical v4 reproduces byte for byte on Linux ARM. A new Linux run of boundary102 was requested with the branch, pinned review commit, command and expected SHA-256. It remains pending; the v4 result is not presented as evidence for the new file.

## Label-free sensor characterization

The preregistered roughness diagnostic uses geometry, clocks and independent periods only. Synthetic calibration recovers 0.5 m iid jitter within 0.02 m. It reads no target labels and changes no model or threshold. Target aggregates stay in ignored local reports. Short-timescale roughness differs substantially from the independent 0.5/2 m jitter probes; this does not identify true coordinate error or exclude smooth/correlated bias. It supplies no automatic model-switch rule.

## Remaining release actions

Wait for independent boundary102 platform evidence and joint team review. Repository publication, final team identity fields and the organizer form are separate pending actions. No organizer submission or public visibility change has occurred. The original final reserve is consumed; further model selection requires a new untouched reserve.
