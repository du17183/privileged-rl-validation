# Phase 3 protocol

Run `bash experiments/door_privileged_ablation/launch.sh` from the project root.
The eight columns of each seed wave run on eight different B300 GPUs.
`diag_A` and `diag_B` retain Phase 2 A/B architecture, training hyperparameters
and seed IDs, with per-vector-step telemetry. Entropy probing samples an action
after each training iteration, so these are independent stochastic reruns and
are not bitwise identical to the frozen Phase 2 A/B runs.
`diag_B` is also B3 (full GT) and B4 (normal full-GT training); no redundant
training is performed. E0/E1 have identical encoder architecture and differ
only by GT prediction weight 0/0.1. The GT head is training-only.

B5 freezes Q after 50k steps but continues actor and entropy optimization.
B6 distills the full-GT Q into a robot-only Q using 1,000 replay minibatches
at 100k steps, then continues ordinary SAC. Distillation costs extra optimizer
work but no additional environment interaction, so report this caveat.
E2 is a targeted follow-up after E1 improved GT decodability but hurt final
success: it uses the same encoder and GT prediction loss only for the first
100k steps, then disables the auxiliary gradient for the remaining 400k.
E3 tests the complementary schedule: it adds GT prediction to the original
3,000 BC updates, then uses ordinary robot-only SAC for all 500k online steps.
The BC minibatch count, expert data and online interaction budget are unchanged.

Each run has 500k online transitions, 5 seeds, 32 environments, 4 optimizer
steps/vector step, 1,000 fixed expert trajectories, and 64 evaluation episodes
every 25k steps. Outputs are isolated under `door_privileged_ablation`.
