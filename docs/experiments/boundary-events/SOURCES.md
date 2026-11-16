# Boundary-event study: evidence and data scope

## Method motivation

[ASRF, WACV 2021](https://yiskw713.github.io/asrf/) separates frame classification from boundary prediction and refines segmentation with boundary evidence. [BaFormer, NeurIPS 2024](https://arxiv.org/abs/2405.15995) also makes boundaries explicit in temporal segmentation. These are conceptual sources for the separate boundary head. This study implements a geometry-based boosted classifier and a duration decoder; it does not import video models or claim the papers' benchmark gains on football.

Previous local studies already tested fixed/adaptive mixtures, longer fitting, short-spell loss weights, temperature/OOD rules and decoder costs. Their failures remain in the reliability/robust-followup reports. The new hypothesis is separately supervised H/A-handover versus live/dead timing, used to vary transition costs over time while retaining frozen class emissions and one-second duration constraints.

## Data availability checked on 16 November

The current [SkillCorner Open Data](https://github.com/SkillCorner/opendata) tree contains 20 match metadata files and 20 phases-of-play references. They are the same twenty A-League games already in the corpus. No additional fresh full-task matches were obtained there.

The [original PFF release](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) covers the known 64 games. Only the existing 36 training games fit the new head; the ten existing development games compare policies. The previous twelve final games are consumed and not scored in this cycle. The six originally excluded bridges remain unopened and unavailable to this study's training/selection code.

The [Signality loader documentation](https://kloppy.pysport.org/user-guide/loading-data/signality/) demonstrates provider loading with small test subsets; this check did not establish a new accessible full-match H/A/D reference set. A supported loader alone is not an eligible independent possession benchmark.

All current evaluation is development research. A development gain does not become a new final PASS without a separately declared untouched confirmation set and frozen single candidate.
