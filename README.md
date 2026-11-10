# Ruses Cules · More Than a Hack 2027

Shared team repository for the Phase 1 ball-possession challenge.

Official challenge and evaluation code: [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027).

## Submission contract

Produce a UTF-8 CSV with header `frame,label`, exactly **199,319** rows and ordered frames **0..199318**. Labels: `H` (home possession), `A` (away possession), `D` (dead ball), `X` (abstain). The ranking metric is **spellF1**; macroF1 is the first tie-break.

The team submission also needs this public source repository and a short technical document covering the approach, assets, training data, possession rules, limitations and reproduction command.

Deadline from the organizer's welcome email: **17 November 2026, 23:59 CET**.

## Working together

- Add the implementation and pinned dependencies, then document one command that regenerates the CSV from the organizer's local data bundle.
- Use feature branches for changes; keep `main` runnable. Check format with the organizer's validator and compare output hashes from clean repeated runs before submission.
- Use the public five-minute ground truth only for pipeline correctness checks. Do not train or tune on it, or use hand labels or external human annotations of the provided game.

## Data stays local

Store organizer materials under `data/` or outside the checkout. Do not commit match video, tracking, tagging, homography, credentials, or derived artifacts that allow reconstruction of the data. The match assets have separate competition-only terms.

Do not upload organizer materials or extracts to hosted LLMs. Run data processing locally or on an authorized team-controlled machine, without exposing those materials to third-party hosted models.

## Before submitting

- Validate the exact team-named CSV with the official validator.
- Confirm byte-identical regeneration from the documented command.
- Finish the team-named technical document with the method actually used.
- Check that only code and permitted documentation are present in the public repository.
- Submit one team response through the organizer's form after joint review.
