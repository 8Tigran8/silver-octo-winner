# Ruses Cules · More Than a Hack 2027

Shared team repository for the Phase 1 ball-possession challenge.

Official challenge and evaluation code: [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027).

## Submission contract

Produce a UTF-8 CSV with header `frame,label`, exactly **199,319** rows and ordered frames **0..199318**. Labels: `H` (home possession), `A` (away possession), `D` (dead ball), `X` (abstain). The ranking metric is **spellF1**; macroF1 is the first tie-break.

The team submission also needs this public source repository and a short technical document covering the approach, assets, training data, possession rules, limitations and reproduction command.

Deadline from the organizer's welcome email: **17 November 2026, 23:59 CET**.

## Install and generate the current v4 CSV

Use Python3.12. V4 uses verified frozen external-trained models included under `assets/v4` (about3.8MB), and predicts onCPU. Generation needs the organizer tactical XML/TXT locally; no external match download or retraining is required for this command.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
MTAH_DATA_DIR=/absolute/path/to/data .venv/bin/python run.py v4 --output 'out/Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py 'out/Ruses Cules.csv'
.venv/bin/python tools/reproduce.py --data-dir /absolute/path/to/data --method v4
```

The data directory contains `tactical/BAR-ATH_tactical_atd.xml` and `.txt`. Only automatic tracking is read; video, homography and SmartTagging are unused. The CLI does not score publicGT unless explicitly requested. The official validator checks format, not accuracy.

The v4 verifier exports committed source twice, independently reads the organizer inputs, verifies frozen asset hashes, predicts onCPU and compares CSV bytes while checking input hashes. This is **frozen-weight inference reproduction**, not a claim that both checks repeat model training. The receipt says which execution mode ran. V1/v2/v3 commands and their earlier raw-retraining receipts remain available.

## Fixed evaluation for future candidates

Use the [evaluation contract](docs/evaluation/README.md) and [machine-readable policy](docs/evaluation/policy.json): compare against frozen v4 on complete whole-game folds; require +0.02 composite spellF1 with source, stress, class and lower-tail guards. Passing development means eligible for confirmation, not final quality PASS. Preserve the unopened PFF country-separated reserve; all prior screened cohorts are development now. The contract explicitly records prospective changes to the older v3/PFF gates without rewriting their history. The [unchanged-v4 control](docs/evaluation/v4_control.json) correctly returns KEEP_V4. No release model or submission CSV changes in this evaluation-policy update.

## Local research after v4 (not published)

[Provider-balanced training](docs/experiments/source-balance/REPORT.md) improves the fixed reused-development composite from 0.61435 to 0.63743, with 29 paired game wins. Reused WCSF and Metrica3 checks improve on natural tracking; Metrica3 with extra gaps falls by 0.00246, and nine European conditional-only scores fall by 0.00206. All registered numeric/provider/reused-source guards pass. A subsequent [all-stage whole-fold audit](docs/experiments/strict-fullfold/REPORT.md) closes inherited isolation: strict paired spellF1 0.61218→0.63557 (+0.02339),29/29paired game wins; versus historical v4,+0.02122 and28/29wins. All numeric/provider guards pass. No fresh independent test has been opened. This is a research candidate, and the default verified release remains v4.

[The prior study](docs/experiments/reliability/REPORT.md) records 19 configurations and [the robustness followup](docs/experiments/robust-followup/REPORT.md) records 14. Their higher-scoring long/OOD candidates were not adopted after Metrica3 regressions. All failures are retained. Provider-balanced weights are frozen locally under `assets/research-balanced`; nothing is submitted, messaged, pushed or published under the user's hold.

## Optional training and research reproduction

The trained components and all source registries are inspectable. To reconstruct the public training inputs (roughly4.5GBraw plus intermediates):

```bash
.venv/bin/python tools/fetch_external.py
.venv/bin/python tools/ext_idsse.py
.venv/bin/python tools/ext_metrica.py
.venv/bin/python tools/fetch_skillcorner.py
.venv/bin/python tools/temporal_trial.py prepare --jobs 3
.venv/bin/python tools/temporal_trial.py fit --split full --device mps
```

The neural training script reads public tracking directly and fits training-only normalization. Use `MTAH_TEMPORAL_CACHE` for a fresh work directory. Research runs used AppleM4MPS; deterministic operations and fixed seeds are recorded. CPU training is available but binary trained weights are not promised identical across backends. The release command uses the frozen checkpoint, independent of training hardware. Original boosted models can be retrained with `run.py v3`; their previously verified source remains documented.

[The technical document](docs/Ruses%20Cules.md) lists every training source, annotation proxy, component, failure mode and reproduction command. [The temporal report](docs/experiments/temporal/REPORT.md) records29-game folds, source-held-out tests, eleven further external games and provenance. Public match assets and intermediates are not committed.

## External quality evidence

The fixed v1 pipeline achieved mean **spellF1 0.6466 versus 0.3244** for the organizer baseline on nine external matches using nested match cross-fitting for both learned stages. It won all nine comparisons; the descriptive paired 95% match-bootstrap interval for the difference is **[0.3062, 0.3400]**. Mean macroF1 was 0.8961 versus 0.7528. The separate Sample_Game_3, absent from the implemented training corpus, scored 0.6837 versus 0.3598 on clean tracking with frozen nine-game weights.

The [full report](docs/experiments/quality-audit/REPORT.md) includes per-match results, decoder/stack/duration ablations, five sensor scenarios, aggregate records, a figure and split provenance. These are external scores: historical configuration selection reused the nine games, sensor corruption is simulated, and Metrica labels are event-derived proxies. The model also predicts about 24% fewer possession spells than the reference. Hidden competition-match quality remains unknown. The audit changed neither the v1 configuration nor the team CSV.

## New-provider confirmation and v2

We added 20 public SkillCorner matches, fixing ten for development and ten for confirmation before scoring. After comparing nine decoder configurations and two sensor-mixture learned candidates, the selected **v2 changes the four transition costs from 6 to 9**. Its learned weights remain v1 weights trained on the original nine games. The sensor-mixture candidates did not pass the development gates and are not used.

On the ten confirmation games, v2 achieved **spellF1 0.5524 versus v1 0.5451**, winning nine games and losing one; paired descriptive 95% gain interval **[0.0044, 0.0100]**. Under extra label-blind corruption: **0.4719 versus 0.4613**, ten wins. On known matched-sensor games there is a small regression, **0.6432 versus 0.6466**. SkillCorner references are automatic phase proxies, with a different sensor/annotation process from the competition. See the [full new-provider report](docs/experiments/skillcorner/REPORT.md) for all candidates, losses, source hashes and the predeclared adoption rule. This supports a modest improvement on the new provider and does not establish hidden-match quality or qualification probability.

## Further team-head experiments

A later [team-ownership study](docs/experiments/team-context/REPORT.md) compared 17 policies: observed ball-contact context, team-only sensor augmentation, matched-head controls and missing-ball gating. The best reused-development composite gain was +0.0036, below the frozen +0.005 replacement threshold. That study retained v2. The later expanded-training study below evaluates the nine additional European matches once for conditional H/A validation; their ownership annotations do not provide reliable D ground truth. No production inference settings changed in this study.

## Expanded team-head candidate v3

V3 expands team-head training to nineteen public games while retaining the v2 dead-ball channel and decoder. Nineteen whole-game exclusions select one fixed 50/50 blend before additional testing. On nine new European games, conditional H/A macroF1 improves **0.8203 → 0.8372**, nine wins, descriptive paired gain interval **[0.0134, 0.0209]**. These references lack D labels, so this is not full-match spellF1.

On the prior ten SkillCorner confirmation games, full spellF1 improves **0.5524 → 0.6046**, ten wins; with added label-blind corruption, **0.4719 → 0.5156**, ten wins. This is a reused secondary cohort, not a pristine new holdout. Known matched-sensor performance changes **0.6432 → 0.6414**. The [expanded-training report](docs/experiments/expanded-team/REPORT.md) includes all four policies, source and split provenance, limitations and predeclared gates. External gates and both clean-source v3 runs passed. These are historical v3 release results; v4 is evaluated below.

## More training data and overfit checks

A subsequent fixed-capacity [19 → 29 game study](docs/experiments/more-data/REPORT.md) added the ten prior confirmation games to research training, excluding every scored game and all its augmentations. Data-only weighted spellF1 gain was +0.00067; descriptive paired 95% interval [-0.00147, +0.00288]. Nine registered decoder/calibration policies then achieved at most +0.00416, below the fixed +0.005 gate. None replaces v3. Baseline v3 and unchanged data29 controls reproduced prior scores exactly. Training/held-out head diagnostics and all candidate results are reported; these reused games do not establish an unbiased final generalization estimate.

A request for the PFF 64-game World Cup2022 dataset is submitted and awaits owner approval. Its prospective split is frozen before gated data/model scores: 36 training, ten development, twelve final-test games with both national teams outside PFF training/development, and six cross-country knockout games excluded to preserve that separation. Source terms, observed-ball availability and event-proxy label semantics still require verification after access. No new PFF score or promotion is claimed; see [the protocol](docs/experiments/more-data/pff_protocol.json).

## Current temporal candidate v4

V4 adds learned25second context and50/50probability fusion to v3. Across29whole-game neural exclusions, composite spellF1 improves0.5657→0.6143 (26wins/3losses; descriptive paired gain interval[0.0434,0.0540]). On a new Groningen/SciSports match:0.5368→0.6858; with extra gaps0.4587→0.5702. Forty screened games are covered in total, including nine conditional-only and reused cohorts; this is not40fresh tests. See the [full report](docs/experiments/temporal/REPORT.md). CPU asset checks match research external scores. Two clean-source CPU generations passed in 25.017 s / 24.239 s and matched preflight byte for byte. An independent raw-data neural refit reproduced the identical checkpoint. See the [inference receipt](docs/experiments/temporal/v4_reproduction.json) and [training repeat](docs/experiments/temporal/neural_reproduction.json). The verified local package is v4; v1/v2/v3 are archived. Submission is a separate pending step.

## Review status

The implementation originates from VV's `model-v1` commit `cc44bfe6bceba39e2e67a03fd32e658cca68bc0e`. The reproduction branch adds setup, explicit output paths, controlled worker counts and verification tooling. V1 remains unchanged. V2 retains its learned models/features and changes only the externally validated transition costs; v3 adds the externally validated nineteen-game team head described above.

Technical v3 verification completed on 13 November: two raw-input clean-source runs passed the official validator and regenerated byte-identical CSVs, matching preflight. Tested source: `6f067004dc59c45ea7e21be6a7bd4246c7e20651`. Timings: 513.040 s and 511.716 s, four workers. CSV SHA-256: `c501145adc08873ae0bab30e88e9cd2cfa88484210c26bed1e6f9fa667724e40`. See the [v3 receipt](docs/experiments/expanded-team/v3_reproduction.json). Competition public GT was not scored.

Historical v2 verification completed on 12 November: two raw-input clean-source v2 runs passed the official validator and regenerated byte-identical CSVs, matching the separate v2 preflight. Tested source: `89737c386949c8dbda69bab95e59b02045194afa`. Timings: 351.340 s and 349.830 s with four workers. The [v2 receipt](docs/experiments/skillcorner/v2_reproduction.json) and technical document record the details. The original v1 remains available and its earlier two clean runs matched VV's supplied hash. Model-selection provenance clarification remains in team review. The external quality audit above provides additional evidence with explicit selection and domain limitations. The repository stays private until the agreed publication window and must be public before submission. The organizer's submission form is a separate release step.

## Data stays local

Store organizer materials under `data/` or outside the checkout. Do not commit match video, tracking, tagging, homography, credentials, or derived artifacts that allow reconstruction of the data. The match assets have separate competition-only terms.

Do not upload organizer materials or extracts to hosted LLMs. Run data processing locally or on an authorized team-controlled machine, without exposing those materials to third-party hosted models.

## Before submitting

- Validate the exact team-named CSV with the official validator.
- Confirm byte-identical regeneration from the documented command.
- Finish the team-named technical document with the method actually used.
- Check that only code and permitted documentation are present in the public repository.
- Submit one team response through the organizer's form after joint review.
