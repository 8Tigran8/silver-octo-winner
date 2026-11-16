# Weighted short-gap candidate

Frozen continuation of boundary102, with the same 65 external training games,
architecture and train-only normalizer. Additional 3,000 steps use training-only
short-possession weights and increased loss weight on missing-ball frames.
The separate activity classifier uses only the original 36 PFF training games.

Inference retains the inherited 50/50 team blend. When the ball is missing,
dead-ball probability is 75% new temporal model and 25% player-activity model.
The selected decoder uses 0.4-second minimum live duration and H/A transition
cost 4.5; dead duration and other transition costs remain unchanged.

The candidate passed the registered ten-game development gates and the
prespecified in-sample regression checks on six training games from three
other providers. Neither is a new independent holdout. The six recently opened
bridge matches and the original twelve final matches were not rescored.

`bundle.json` identifies all model/metadata hashes. Training provenance is in
`neural_meta.json` and `activity_meta.json`. No raw tracking, reference arrays,
training features or organizer labels are included. See the full experiments
under `docs/experiments/spell-weighted-finetune` and `short-gap-recovery`.
