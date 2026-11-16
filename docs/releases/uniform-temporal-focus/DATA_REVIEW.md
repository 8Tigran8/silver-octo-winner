# Data coverage review — 17 November 2026

We have not exhausted all potentially useful football data. This review checks source inventories and documentation; it does not add training matches, open the 18 audit references, or evaluate the target labels. Inventory evidence is in `data_inventory_check.json`.

## What is already covered

| Source | Relevant current inventory | Existing use | Fresh independent matches in this release? |
|---|---|---|---|
| Original PFF 2022 World Cup | 64 matches | 36 train, 10 repeatedly used development, 18 previously opened audit | No |
| SkillCorner open data | Actual matches.json lists 20 | All 20 in training | No |
| DFL/IDSSE | 7 tracking/event matches | All 7 in training | No |
| Metrica sample data | Games 1, 2, 3 | 1/2 train; 3 previously investigated | No |

The 65-game training union includes every inherited boosted-model training game: the older nine-game model uses seven DFL and two Metrica games; the 19-game team model adds ten of the same 20 SkillCorner games (`mtah/expanded_team_training.json`). The new specialists do not introduce extra matches. Training scenario augmentation does not increase independent match count.

PFF's original provider describes a [64-game release](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data); its Hugging Face mirror is the same World Cup, not another validation cohort. [SkillCorner's actual manifest](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json) is more current than its ten-game README description: all 20 IDs match our registry. New body-pose files for two existing matches are additional features, not new games. [DFL's paper](https://www.nature.com/articles/s41597-025-04505-y) describes seven full games with synchronized tracking/events, consistent with the actual downloaded-file inventory. [Metrica's sample repository](https://github.com/metrica-sports/sample-data) does not establish another unused full-match cohort.

## Most promising unused source: SoccerTrack v2

[SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2) provides ten full university matches (934 minutes), player/team positions and action annotations. Its [dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2) lists CC BY 4.0 data and gated access. We have not accepted that gate or downloaded the raw matches. Pin v1.2 or later with the documented Match 10 alignment fix.

Crucially, the supplied ball trajectory is reconstructed between annotated event anchors. If the possession reference is reconstructed from those same events, using that ball trajectory as model input creates a shared-label dependency and can inflate apparent accuracy. The BAS annotations are action events, not a ready-made continuous H/A/D possession reference. Ten new match IDs alone do not establish a clean Spell F1 test.

This source is worth pursuing, but only after checking raw signal provenance, timeline alignment and possession semantics. The provider raw exports may help; their ball independence is not yet verified. The [GSR format documentation](https://raw.githubusercontent.com/AtomScott/SoccerTrack-v2/main/docs/format-gsr.md) also warns that shipped schema and declared image dimensions differ from the written specification, so an adapter must be checked against actual files. Independent video-derived ball tracking and independently annotated possession would support a stronger test. A player-only experiment would answer a narrower question and must be reported separately from the full production model.

## Other sources and why counts can mislead

[SoccerNet-GSR](https://arxiv.org/abs/2404.11335) supplies 200 annotated 30-second clips for player localization/identification, not 200 full matches with continuous possession. It can help a tracking subsystem but cannot directly replace our full-match H/A/D benchmark. [StatsBomb open data](https://github.com/hudl/open-data) supplies events and selected 360 snapshots; those are not continuous ball/player trajectories. Extra event-only games require independently matched tracking before they can validate this inference pipeline.

## Recommended next evaluation, without contaminating the release

1. Keep this release frozen. Register all new match IDs and deduplicate by teams/date/competition before reading labels or scores.
2. Verify provenance and licensing, coordinate units, period boundaries and event-to-frame synchronization. Do not derive predictor geometry from the labels used as the reference.
3. Reserve whole matches, preferably an entire new competition/provider, for a single prospective final test. Separate any adapter-development matches first; freeze reference-construction rules before scoring.
4. Compare the frozen old/new versions on exactly the same valid frames using the official metric, per-match differences, match-level uncertainty, worst regressions, short-spell recall and missing-ball team/dead errors. Frames and synthetic seeds are not independent samples.
5. If labels influence a correction or model choice, that cohort becomes development. A new final claim then needs another untouched cohort.

Expanding independent validation has higher evidential value now than moving known test games into training. It could reveal a weakness even if the mean score falls; that is useful information. More training games may improve quality, but no gain is guaranteed. Repeated development success and technical reproducibility do not prove absence of overfitting or a place in the final.
