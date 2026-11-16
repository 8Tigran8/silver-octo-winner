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

At 11:43 CET on 16 November, VV reported two matching boundary102 runs on Linux ARM (aarch64), Python 3.12.3 and Torch 2.8.0 CPU; both official format checks passed. The reported CSV SHA-256 equals our frozen candidate. He used the archive of review commit `072feab`, initialized as a local Git repository, and reports matching code/weight/data fingerprints. This is teammate-reported archive-source evidence; the original machine receipt was not imported or independently audited here. It is not a fresh upstream-clone or Linux/x86 claim. See [the scoped report](experiments/pff-mixed/linux_arm_report.json).

## Label-free sensor characterization

The preregistered roughness diagnostic uses geometry, clocks and independent periods only. Synthetic calibration recovers 0.5 m iid jitter within 0.02 m. It reads no target labels and changes no model or threshold. Target aggregates stay in ignored local reports. Short-timescale roughness differs substantially from the independent 0.5/2 m jitter probes; this does not identify true coordinate error or exclude smooth/correlated bias. It supplies no automatic model-switch rule.

## Remaining release actions

Complete joint team review; Linux ARM reproduction is now reported with the evidence limitations above. Repository publication, final team identity fields and the organizer form are separate pending actions. No organizer submission or public visibility change has occurred. The original final reserve is consumed; further model selection requires a new untouched reserve.
