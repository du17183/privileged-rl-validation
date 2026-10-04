# Drawer protocol development record

## Archived first full protocol (`results/v1/`)

Eight paired A/B/C seeds each completed 200,000 online SAC steps with 500 clean successful demonstrations. A/B final success was zero across all seeds; C reached only 1/64 episodes for one seed. No 50% threshold was reached, so this protocol could not test the fixture-state hypothesis. Its checkpoints, evaluation CSVs, curves, and report are retained under `checkpoints/v1/`, `results/v1/`, and `docs/rl_experiment_report_v1.md`.

The physical diagnosis found BC often closed the gripper too far from the handle and SAC then moved away. Investigation also found Isaac Lab's automatic reset could expose frame-transformer poses from the preceding episode before a physics forward. The environment now forwards physics after automatic reset; a forced-reset test confirms TCP and handle frames match a fresh explicit reset. The robot-side observation also includes the reset-controller clock. These changes define a new protocol; v1 and current results must not be pooled.

## Current primary protocol

Two planner collections yielded 1,000 successful demonstrations (500 clean, 500 with closed-loop correction after arm-action perturbation) from 1,012 attempts. A/B/C use identical BC, offline critic preparation, replay, reward, action, and online budgets; A/B actors begin with exactly equal weights for every paired seed. The only A/B distinction is whether the critic receives the 11-D fixture state. Settings are frozen in `configs/experiment_primary.json`.

Pilot seeds 89 and 90 showed that a privileged critic can produce genuine bilateral handle contact and 64/64 drawer success at some checkpoints, while the robot-only baseline lagged. Policies were sometimes unstable and success later fell. These pilot seeds are excluded from the formal eight-seed statistics and saved in `results/pilots/`.

## Separate corrective-label diagnostic

A DAgger-style rollout collected 100,000 additional simulator interactions with planner labels in `datasets/drawer_dagger_v1.h5`. Ten thousand BC updates using these labels produced strong grasping in several pilot seeds, including a seed where A/B both started at 100% success. That saturation makes it unsuitable as the primary sample-efficiency test. This dataset is excluded from the primary training protocol; any future use must include its 100,000 interactions in the conditioning budget.
