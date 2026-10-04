# Panda Drawer privileged-information RL validation

This project tests whether fixture ground truth can improve online SAC sample efficiency for a Franka Panda opening an articulated drawer. The primary paired experiment runs A (robot-only actor and critic), B (robot-only actor, privileged critic), and C (privileged actor and critic) on eight seeds. The A/B evaluation actor has no fixture-state input.

## Server and simulator

- Workspace: `/home/xiaolong/privileged_rl_validation` on b300-2.
- Eight NVIDIA B300 GPUs; Isaac Sim 5.1, Isaac Lab v2.3.0, PyTorch 2.7.0+cu128.
- The task adapts `Isaac-Open-Drawer-Franka-IK-Rel-v0` for the Panda and Sektion cabinet. It uses a 7-D relative IK plus gripper action, 0.30 m drawer success threshold, filtered finger-handle contact, automatic reset, and one reward for all variants.
- `envs/isaac_env.py` exposes `get_tool_state()` as the future fixture interface. The A/B actor consumes 26 robot-side signals including reset-controller time; the critic B additionally consumes 11 drawer/handle/contact values.
- GPU compatibility and complete package versions: `docs/environment_setup.md`. Reset, task, and data protocol: `docs/drawer_task_report.md`.

## Data

`datasets/drawer_expert_1000.h5` contains 1,000 successful planner/IK demonstrations: 500 clean and 500 with action perturbations corrected in closed loop. Collection required 1,012 attempts and 308,256 simulator interactions; 300,645 transitions are saved. Each trajectory has observation, fixture state, action, reward, next observation/state, done, terminated, and truncated. Audit: `docs/dataset_report_v2.md`.

`datasets/drawer_dagger_v1.h5` contains a separate 100,000-interaction corrective-label diagnostic. It is **not** used by the primary protocol. Preliminary results and the first failed full protocol are retained under `results/pilots/` and `results/v1/`.

## Primary experiment

The locked settings are in `configs/experiment_primary.json` and `train/launch_all.sh`: seeds 0–7, one seed per GPU, A/B/C each with 200,000 online environment interactions, 32 parallel environments, 3,000 BC updates, 1,000 offline critic updates, four SAC updates per vector step, 25% expert replay, constant actor BC regularization weight 10, and 64 evaluation episodes at step zero and every 10,000 online steps.

Launch from the project root with `bash train/start_all.sh`. Check `logs/launch_all.exit_code` for completion; 0 means all 24 runs passed `evaluation/verify_runs.py` and aggregation. Per-run TensorBoard events and stdout are under `logs/`, checkpoints under `checkpoints/`, and evaluation CSVs/plots/statistics under `results/`. `results/thresholds.csv` reports 50/80/90% crossings both in online steps and total interactions including evaluation and the shared expert-collection budget.

The completed eight-seed comparison did **not** establish a privileged-critic gain: B−A normalized success-AUC difference −0.0208, exact paired p=0.7969. Policies often reached high success and later regressed. The full result, thresholds, limitations, and contact checks are in `docs/final_report_zh.md` and `results/rl_experiment_report.md`.

After the primary experiment, `bash evaluation/run_heldout.sh` independently replayed all 24 final checkpoints for 64 episodes each with reset seed 901. Its 670,144 additional simulator interactions and per-seed results are in `results/heldout_final_summary.json` and `results/heldout_final.csv`; they are separate from the primary learning curves.

Only the Drawer task is implemented. Door articulation and DIVL are planned extensions after Drawer validation.

