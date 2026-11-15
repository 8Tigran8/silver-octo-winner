# Frozen boundary102 snapshot

This is the single candidate selected before opening the original twelve-game PFF final reserve. Final validation and release status are recorded in the experiment report, separately from this immutable snapshot.

The temporal branch trains on65public games: the original29 plus36original-provider PFF World Cup matches. Its108geometry/contact features and108missing masks enter48channels and seven residual blocks (dilations1/2/4/8/16/32/64), giving101.8seconds of convolutional context. Training uses6000AdamWupdates, seed20261113, equal sampling of four providers and a fixed0.15boundary auxiliary loss. Full metadata includes source hashes, training members, sequence hashes and the train-only normalization.

Inference needs this bundle and the unchanged `assets/v4` components. It uses their v3 base probabilities, averages them50/50with the new temporal branch, and retains the cost9duration decoder. It needs no training cache, external match download or ground-truth file. PyTorch loads the new checkpoint with `weights_only=True`; model/metadata/baseline hashes are checked.

PFF features use raw VISIBLE observations only; event-informed smoothed coordinates are excluded. Raw source data and reference arrays are not redistributed. See the registered study and source manifests under `docs/experiments/pff-mixed` and `tools/pff_source_files.json`.
