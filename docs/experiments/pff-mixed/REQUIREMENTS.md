# Independent check against the original challenge brief

Reviewed 15 November 2026 against complete original RULES.md, SUBMISSION.md and EVALUATION.md retrieved at 2026-11-15T20:00:48.379024+00:00. The raw files remain preserved locally under reports/release_review/official; this matrix does not replace them.

## Original source identities

- [RULES.md](https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/main/RULES.md): SHA-256 `b6f63456d215e2e6016619f6268580ed8921bb96ba41c7d3037f2733f8d10153`.
- [SUBMISSION.md](https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/main/SUBMISSION.md): SHA-256 `acb90eebc0f113fc072d90ac77be501e63de1dde65e9227186b4718bc24f22d2`.
- [EVALUATION.md](https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/main/EVALUATION.md): SHA-256 `25b61f6d8a2bcbe77396bd9a9cae564b27424538e0a958aa1cd365e1e58ae571`.

## Requirement coverage

| Original requirement | Expected evidence | Visible location | Verification |
|---|---|---|---|
| SUBMISSION: header, exactly 199,319 ordered frames and H/A/D/X | Official validator on exact review CSV | PDF p1; CSV | PASS |
| SUBMISSION: approach and why | Boosted geometry + longer temporal context + duration decoder | PDF p1 | PASS |
| SUBMISSION: which supplied assets used or dropped | Tactical XML/TXT used; video, homography, Smart Tagging unused | PDF p1 | PASS |
| SUBMISSION: every labelled dataset, source, who and how much | Provider label/proxy table, 65 games, per-provider reference-frame counts, 9/19 inherited registries, validation-only sources | PDF p2 | PASS: provider workflows explicitly not independently audited |
| SUBMISSION: loose balls, challenges, restarts | Learned contact/context; no every-touch turnover; stoppages split possessions; restart snapping disabled | PDF p1 | PASS |
| SUBMISSION: limitations and fixes | Short-spell undersegmentation, Metrica train/CV gap, proxy labels, offline future context, new-reserve requirement | PDF p3-4 | PASS |
| SUBMISSION/RULES: exact reproduction | Pinned dependencies, assets, run command; two clean CPU runs with identical SHA-256 | PDF p4; reproduction.json | PASS: inference, not repeat training |
| RULES: algorithmic, no hand annotation/correction of provided game | Entire output from frozen code; no manual frame overrides | PDF p1-2; generator and frozen source | PASS |
| RULES: no private GT or human-produced target labels, no public-window tuning | No target reference enters training/selection/calibration; no target quality score | PDF p1-2; split/source registries | PASS |
| RULES: organizers can inspect/run every component | Included model weights; accessible public training sources and original PFF request route; explicit sources | PDF p2 and p4; repository | LOCAL PASS; source publication held |
| RULES: do not redistribute organizer assets or reconstructive data | Raw tracking/video/reference arrays excluded; tracked history audited; aggregate-only receipts | PDF p4; .gitignore; local history audit | PASS: bounded payload audit |
| EVALUATION: spellF1 ranking, macro/class guard, unchanged scorer | Pinned official metric; same-game comparisons; all predeclared guards; final consumed once | PDF p3; final_results.json and final_freeze.json | PASS |
| SUBMISSION: short technical document and team filenames | Four-page PDF plus exact team CSV, both under 10 MB | out/review-ready | PASS |
| SUBMISSION: public repository URL | Current source is committed locally; remote remains private | README and PDF p4 URL | PENDING: publication on explicit user hold |
| SUBMISSION: one team form response, member identity fields | No form submission or outbound message made | Local package only | PENDING: user hold; names intentionally deferred |
| RULES: 3-5 registered human team members for later stages | Registration membership must be confirmed separately; AI is not a member | Organizer registration/form | NOT independently verified; no eligibility claim |

Technical quality status: **PASS within the declared evaluation contract**. Local review artifacts are complete. This does not certify global absence of overfit, hidden competition performance, team eligibility or organizer submission.

Publication, form submission and messages remain on user hold. The original 12-game final is consumed; no changed candidate may be selected using it.

The largest material weakness is the standalone neural Metrica full-fit/whole-fold-out gap (0.8565 vs 0.7056 on two games). Independent full-pipeline final improvement is separately positive on 12/12 games. Both facts remain visible in the technical document.

Supplemental update, 16 November: the two-game synthetic sensor diagnostic reveals material coordinate-jitter regressions. These are explicitly visible in PDF page 4, the current technical Markdown and the linked supplemental report; the original fixed final PASS does not certify arbitrary sensor robustness.
