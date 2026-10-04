# Phase 6: stable expert-to-online SAC

## Fixed setup

The task, reward, reset, Panda robot, Isaac Lab environment, and the original
1,000 successful expert trajectories are read from the existing project without
editing them. All Phase 6 outputs use separate `phase6_stable_online_rl`
subdirectories. The deployment actor and the SAC critic both receive only the
26-dimensional robot observation. The 11-dimensional tool state is stored for
audit and potential later work, but is never an actor or critic input here.

Each unique arm uses seeds 0–4, 32 parallel environments, 500,000 **training**
interactions, four SAC optimizer updates per vector step, minibatch 256, and
an independent, fixed-reset evaluation of 64 episodes at step 0 and every
10,000 training steps. Evaluation interactions are counted separately.

| Label | Expert replay fraction per SAC minibatch | Actor BC warm start | Online LR | Actor BC weight |
|---|---:|---|---:|---:|
| A0 | 0% | No | 3e-4 | 0 |
| A1 | 70% | No | 3e-4 | 0 |
| A2 = B0 | 50% | No | 3e-4 | 0 |
| B1 | 50% | Yes, 3,000 updates | 3e-4 | 0 |
| B2 | 50% | Yes, 3,000 updates | 1e-4 | 0 |
| B3 | 50% | Yes, 3,000 updates | 3e-4 | 10 |

A0–A2 isolate expert replay. A2/B0–B3 isolate actor BC initialization and
the subsequent update constraint. All arms start with a fresh random SAC
critic; **offline critic warmup is zero in every arm**. Consequently A0 is
genuinely online-only SAC, and B0 is the random-initialization control for
50% expert replay. B0 reuses A2 rather than spending another 2.5 million
interactions on an identical arm.

After BC, B1/B2/B3 reset the actor Adam optimizer before online learning.
This makes the BC comparison about actor weights rather than inherited BC
optimizer momentum; B2 then applies its lower online learning rate.

`offline_rl/baseline.py` also fits BC on 800 successful expert trajectories and
reports action MSE on 200 disjoint expert trajectories for five seeds. This
diagnostic has zero online interaction. B1 step-0 checkpoints are independently
evaluated to measure actual BC control before SAC updates.

## Reproduction

```bash
cd /home/xiaolong/privileged_rl_validation
source configs/runtime_env.sh
bash experiments/phase6_stable_online_rl/launch.sh
bash experiments/phase6_stable_online_rl/launch_heldout.sh
.venv/bin/python experiments/phase6_stable_online_rl/verify_protocol.py
.venv/bin/python experiments/phase6_stable_online_rl/analyze.py
.venv/bin/python offline_rl/baseline.py --device cpu --updates 3000
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/phase6_stable_online_rl/reset_audit.py --device cuda:0
bash experiments/phase6_stable_online_rl/launch_noise_sweep.sh
.venv/bin/python experiments/phase6_stable_online_rl/analyze_noise_sweep.py
```

The analyzer refuses to aggregate incomplete runs. It reports across-seed
Student-t 95% intervals and exact two-sided paired sign-flip tests (32 sign
assignments). With five seeds, the smallest possible exact two-sided p value
is 0.0625; a directional trend alone is not proof of p<0.05 significance.
Best checkpoints are selected by the fixed evaluation curve and then tested
in a separate process with a different evaluation RNG seed. A read-only reset
audit found just **one distinct initial robot/tool state across 32 environments**
at 1e-5 precision for both RNG seeds 50000 and 90000. Thus the second process
checks repeatability on the same nominal fixture state, not generalization to
different initial poses. Selection-curve best-minus-final is vulnerable to
winner's bias; the independent-process counterpart is a narrower, fixed-state
deployment estimate. Evaluation episodes should not be treated as 64
independent task configurations.

The noise sweep replays the saved B3 best actor without optimizer updates. It
caps the pre-tanh Gaussian action standard deviation at 0.135, 0.05, or 0.01
only during evaluation. It tests execution sensitivity; it does not establish
that training with a cap improves online learning. The fixed-reset audit,
training/evaluation interaction counts, and this distinction are included in
`docs/phase6_stable_rl_report.md`.
