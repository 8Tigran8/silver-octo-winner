# PFF World Cup 2022: source, access and reference labels

Updated 16 November 2026. The training source is the original PFF FC World Cup 2022 release, obtained on 15 November through the [provider's access form](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset). The article describes broadcast tracking, event data, metadata and specifications for 64 games. Submitting its form provided access to the original provider folder. Organizers can use that same request route; this is not an unavailable private in-house corpus.

## Terms and redistribution

The original article does not state a standard dataset license. We do not infer a permissive redistribution grant from the availability of the download. Original raw tracking, event/reference arrays, rosters and vendor specification PDFs are kept outside Git. The included material is source code, frozen trained weights and aggregate evaluation/provenance receipts.

The separate [OpenSportsLab/PFF Hugging Face mirror](https://huggingface.co/datasets/OpenSportsLab/PFF) is marked GPL-3.0 and gated. Its account request remains pending and the mirror was not used. That mirror label is not treated as evidence that GPL-3.0 applies to the original provider release.

## What is used and how labels are derived

The neural branch trains on the original 36 registered PFF training games. Ten development games are used for selection; the original twelve final games were opened once after a single candidate was frozen. Six cross-country bridges remain excluded. The complete IDs and country separation are in the [original split](experiments/more-data/pff_protocol.json).

Predictors use only raw VISIBLE player/ball geometry, clocks and independent periods. ESTIMATED tracks, event-informed smoothed coordinates, event fields, possession indicators and player/team identities do not enter features.

Reference H/A/D arrays are reconstructed separately from embedded provider events using the pinned primary kloppy state-machine semantics: OUT/END indicates dead ball; kickoff/OTB indicates live play; home_ball supplies ownership. Periods reset state, contradictory duplicates are masked, and a fixed SDK last-row shadow checks sensitivity. These are provider-event-derived approximations, not manually corrected frame-by-frame competition truth. Provider annotation workflows were not independently audited.

The [source adaptation record](experiments/pff/REPORT.md) and [65-game report](experiments/pff-mixed/REPORT.md) preserve source hashes, loader revision, checks, label coverage and failed precursor experiments. Vendor PDFs remain local because they carry proprietary notices.

## Reproduction

Generating the organizer CSV uses the bundled frozen model and requires no PFF access or training download. Optional training reproduction requires obtaining the original files through the provider, then using the pinned manifests and the training commands in [README.md](../README.md). No provider form is submitted by the inference command.
