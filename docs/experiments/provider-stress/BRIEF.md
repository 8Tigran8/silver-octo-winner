# Provider/sensor stress pilot

14 November 2026. Collaboration brief; no new model or score yet.
This is a diagnostic addition to the fixed evaluation contract, not a change to
its 29-game/67-cell ranking benchmark or its unopened PFF reserve.

## VV: build a reproducible pilot on two existing development games

Use one original DFL game and one development SkillCorner game already downloaded.
Keep original input files intact. Provide a script plus a JSON configuration with
source IDs/hashes, random seed, source sampling rate, noise and ball-gap parameters.
Start with 10/15 Hz observation rates, coordinate noise between 0.5 and 2 metres,
and multi-second ball-dropout bursts. Record chosen presets before model scores;
keep development experiments and failure cases visible. We can then apply the same
presets to the rest of the fixed development corpus.

- Run perturbation from geometry, clocks and independently supplied period metadata.
  Do not read possession, alive/dead, event or reference-coverage fields to decide
  where/how to corrupt. Keep reference labels separate and copy them unchanged.
- Preserve observed gaps; do not turn an interpolated position into a measured
  ball observation. Do not corrupt across a half boundary. Keep half metadata.
- Original game and all corrupted/team-swapped copies remain in the same fold.
  Score with that game's excluded-fold model, not the full 29-game release fit.
- Altered versions of familiar games test sensor robustness. They are not an
  independent new-provider dataset or extra independent matches.

## Interchange format

Use the same numeric NPZ schema as `tools/ext_idsse.py`/`tools/skillcorner_reader.py`:

| Key | Shape / meaning |
|---|---|
| `players` | float `[N,S,3]`; normalized pitch x/y, speed in metres/second; missing observations NaN |
| `ball` | float `[N,3]`; normalized pitch x/y, height in metres or NaN |
| `slot_team` | int `[S]`; 0 home, 1 away; slots represent track fragments |
| `pitch` | `[length,width]` in metres |
| `period` | int `[N]`, from source clock/metadata, 0 outside play periods, 1/2 halves |
| `gt` | int `[N]`; H=0, A=1, D=2, X=3, unchanged reference, never predictor input |

Our feature code expects a canonical **25 Hz** time grid. For 10/15 Hz sensor
experiments, supply native observation timestamps and the mapping to that grid in
the manifest. Do not silently reinterpret 10 Hz rows as 25 Hz frames. Preserve
missingness when mapping observations; provide any resampling rule in code. Compute
speeds with physical elapsed time within a period, without crossing missing tracks.
Reference remapping must follow source clock alignment, never model predictions.

Commit the converter/configuration only. Keep NPZs and raw inputs in ignored local
folders. Include a check that changing only reference labels leaves corrupted
geometry, timestamps and masks identical for the same seed. When the pilot arrives,
we check schema/timing/label independence before any model comparison.

## Tigran: candidate comparison

Keep v4 as the verified baseline. The next proposed model experiment replaces the
constant 50/50 mixture with a learned reliability mixture conditioned on observation
gaps and model uncertainty. It must use training-only/out-of-fold signals for
fitting; scoring a gate on probabilities fitted to its own test games is leakage.
Use the existing prospective quality gates and predeclared seed policy. The stress
pilot is additional diagnosis; it cannot overwrite frozen benchmark scenarios.

## SoccerTrack v2: eligibility review only

Primary sources inspected on 14 November:
[dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2) and
[author repository](https://github.com/AtomScott/SoccerTrack-v2).

The card describes ten university matches, player state/action annotations, and
an auxiliary ball trajectory reconstructed from annotated event anchors using
constant-speed straight-line interpolation. That trajectory carries event timing.
Using it as predictor input while deriving possession targets from the same events
would make an independent possession evaluation circular. Treat any such score as
ineligible evidence of measured-ball generalization.

Independent player geometry may still support an auxiliary experiment, provided we
validate reference ownership/dead-ball semantics and never use event-derived ball
or event fields as predictors. Full-frame H/A/D eligibility has not been established.
No files downloaded, no model scores inspected, no training performed. The HF card
is gated and asks for account information sharing; no access request or agreement
has been submitted for this new source. Dataset card says CC BY 4.0; inspect actual
access terms before requesting files. Existing PFF authorization is separate.
