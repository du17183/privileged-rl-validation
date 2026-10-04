# Panda Drawer BC → online SAC experiment

Eight paired seeds (0–7) run on eight B300 GPUs. Each A/B/C run uses the same 500 successful demonstrations, BC budget, reward, 32 parallel environments, 200,000 online training interactions, 10,000-step evaluation schedule, and 64 episodes per checkpoint. Evaluation interactions are logged separately from training steps. A has robot-only actor/critic; B has robot-only actor and robot+fixture critic; C has robot+fixture actor/critic and is an upper bound rather than the deployment policy.

| Variant | Final success mean ± SD | Success AUC mean ± SD | Wall time mean |
|:--|--:|--:|--:|
| A | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.12 h |
| B | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.13 h |
| C | 0.002 ± 0.006 | 0.000 ± 0.000 | 0.13 h |

## Sample-efficiency thresholds

Values below are median training steps across seeds that reached the threshold; `not reached` means no seed reached it. Individual crossing points, total interactions including evaluation, and wall times are in `results/thresholds.csv`.

| Variant | 50% | 80% | 90% |
|:--|--:|--:|--:|
| A | not reached | not reached | not reached |
| B | not reached | not reached | not reached |
| C | not reached | not reached | not reached |

## Paired A/B comparison

B minus A mean normalized success-AUC gain: 0.0000; exact two-sided paired sign-flip p = 1.0000 across 8 matched seeds.

## Conclusion

No variant achieved even 50% final success. This run does not establish whether fixture ground truth improves online RL sample efficiency; the shared BC/SAC recipe or observation design must first be made capable of solving the task.

Figures: `results/success.png` and `results/reward.png`. Per-run CSVs, aggregate curves, thresholds, TensorBoard events, final checkpoints, and the exact experiment configuration are retained in the project.
