# Panda Drawer BC → online SAC experiment

Eight paired seeds (0–7) run on eight B300 GPUs. Each A/B/C run uses the same successful planner demonstrations, BC budget, reward, 32 parallel environments, 200,000 online training interactions, 10,000-step evaluation schedule, and 64 episodes per checkpoint. Evaluation interactions are logged separately from training steps. A has robot-only actor/critic; B has robot-only actor and robot+fixture critic; C has robot+fixture actor/critic and is an upper bound rather than the deployment policy.

Shared pretraining simulator interactions: 308,256. These are added to the total-interaction threshold column but are not multiplied by the number of seeds.
A/B actor parameters and BC-only evaluation match exactly within every paired seed; `evaluation/verify_runs.py` checks this before reporting results.

| Variant | Final drawer success mean ± SD | Final contact-assisted success mean | Peak success mean | Success AUC mean ± SD | Wall time mean |
|:--|--:|--:|--:|--:|--:|
| A | 0.242 ± 0.449 | 0.242 | 0.754 | 0.233 ± 0.246 | 0.13 h |
| B | 0.254 ± 0.461 | 0.254 | 0.961 | 0.213 ± 0.149 | 0.13 h |
| C | 0.484 ± 0.519 | 0.484 | 0.961 | 0.291 ± 0.208 | 0.13 h |

## Sample-efficiency thresholds

Values below are median online training steps across seeds that reached the threshold; `not reached` means no seed reached it. Individual crossing points, total interactions including shared pretraining and evaluation, and wall times are in `results/thresholds.csv`.

| Variant | 50% | 80% | 90% |
|:--|--:|--:|--:|
| A | 50000 (6/8 seeds) | 50000 (6/8 seeds) | 50000 (6/8 seeds) |
| B | 50016 (8/8 seeds) | 50016 (7/8 seeds) | 50016 (7/8 seeds) |
| C | 25008 (8/8 seeds) | 60000 (7/8 seeds) | 60000 (7/8 seeds) |

## Thresholds retained through the final evaluation

The first listed checkpoint and every later checkpoint meet the threshold; at least two checkpoints are required.

| Variant | 50% | 80% | 90% |
|:--|--:|--:|--:|
| A | 120000 (1/8 seeds) | 120000 (1/8 seeds) | not reached |
| B | 170016 (1/8 seeds) | 170016 (1/8 seeds) | 170016 (1/8 seeds) |
| C | 160016 (2/8 seeds) | 160016 (2/8 seeds) | 160016 (2/8 seeds) |

## Paired A/B comparison

B minus A mean normalized success-AUC gain: -0.0208; paired bootstrap 95% interval [-0.1926, 0.1459]; exact two-sided paired sign-flip p = 0.7969 across 8 matched seeds.

## Independent final-checkpoint replay

Each of 24 final checkpoints ran 64 episodes with reset seed 901, adding 670,144 evaluation interactions outside the periodic curves.

| Variant | Held-out final success | Contact-assisted success |
|:--|--:|--:|
| A | 0.246 | 0.246 |
| B | 0.270 | 0.270 |
| C | 0.488 | 0.488 |

B minus A held-out final-success mean gain: 0.0234; exact paired sign-flip p = 0.7500.

## Conclusion

The completed runs do not provide statistically supported evidence that the privileged critic improves success AUC under this protocol. Inspect per-seed threshold crossings and learning curves before interpreting the sign of the effect.

Figures: `results/success.png` and `results/reward.png`. Per-run CSVs, aggregate curves, thresholds, TensorBoard events, final checkpoints, and the exact experiment configuration are retained in the project.
