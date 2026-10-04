"""Collect Door episodes from existing frozen Actor checkpoints; no RL update."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("E0", "E25", "E100", "E100R1", "Q0"), required=True)
parser.add_argument("--mode", choices=("best", "final"), required=True)
parser.add_argument("--seed", type=int, choices=range(5), required=True)
parser.add_argument("--episodes", type=int, default=125)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401

from auxiliary_learning.gt_prediction import EncodedGaussianActor
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env, transition_after_step

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "frozen_rollouts"


def selected_checkpoint():
    path = ROOT / "results" / "stable_privileged_rl" / f"eval_{args.variant}_seed{args.seed}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = max(rows, key=lambda r: float(r["success_rate"])) if args.mode == "best" else rows[-1]
    step = int(row["env_steps"])
    ckpt = (ROOT / "checkpoints" / "stable_privileged_rl" /
            f"{args.variant}_seed{args.seed}" / f"step_{step}.pt")
    return ckpt, step


@torch.no_grad()
def main():
    if args.episodes != 125:
        raise ValueError("Phase 5 split requires exactly 125 episodes per frozen policy")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.variant}_{args.mode}_seed{args.seed}"
    output = OUT / f"{name}.h5"
    if output.exists():
        with h5py.File(output, "r") as h5:
            if len(h5) == 125 and bool(h5.attrs.get("complete", False)):
                print(f"Already complete {output}", flush=True)
                return
        raise RuntimeError(f"Incomplete existing rollout file: {output}")
    ckpt, step = selected_checkpoint()
    reset_seed = 200000 + args.seed
    cfg = make_cfg(32, device=args.device or "cuda:0", seed=reset_seed)
    env = create_env(cfg)
    try:
        actor = EncodedGaussianActor(26, 7).to(env.device)
        actor.load_state_dict(torch.load(ckpt, map_location=env.device, weights_only=False)["actor"])
        actor.eval()
        env.reset(seed=reset_seed)
        counts = np.zeros(env.num_envs, dtype=np.int64)
        pending = [[] for _ in range(env.num_envs)]
        episodes = {}
        fields = ("observation", "state", "action", "reward", "next_observation",
                  "next_state", "done")
        for _ in range(2500):
            robot = robot_observation(env)
            gt = env.get_tool_state()
            action, _ = actor(robot, deterministic=True)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            arrays = (robot.cpu().numpy(), gt.cpu().numpy(), action.cpu().numpy(),
                      reward.cpu().numpy().reshape(-1, 1), next_robot.cpu().numpy(),
                      next_gt.cpu().numpy(), done.cpu().numpy().reshape(-1, 1))
            done_cpu = done.cpu().numpy()
            for env_id in range(env.num_envs):
                if counts[env_id] >= 4:
                    continue
                pending[env_id].append(tuple(array[env_id].copy() for array in arrays))
                if not done_cpu[env_id]:
                    continue
                slot = int(counts[env_id] * env.num_envs + env_id)
                if slot < args.episodes:
                    trajectory = {field: np.stack([step_row[j] for step_row in pending[env_id]])
                                  for j, field in enumerate(fields)}
                    rewards = trajectory["reward"].reshape(-1)
                    mc = np.empty_like(rewards)
                    running = 0.0
                    for t in range(len(rewards) - 1, -1, -1):
                        running = float(rewards[t]) + 0.99 * running
                        mc[t] = running
                    trajectory["mc_return_to_go"] = mc[:, None]
                    trajectory["success"] = bool(env.final_door_angle[env_id, 0] > SUCCESS_ANGLE_RAD)
                    trajectory["episode_return"] = float(rewards.sum())
                    trajectory["final_angle_rad"] = float(env.final_door_angle[env_id, 0])
                    episodes[slot] = trajectory
                pending[env_id] = []
                counts[env_id] += 1
            if bool(np.all(counts >= 4)):
                break
        if len(episodes) != args.episodes:
            raise RuntimeError(f"Collected {len(episodes)}/{args.episodes} complete episodes")
        temp = output.with_suffix(".partial.h5")
        with h5py.File(temp, "w") as h5:
            h5.attrs.update({"complete": True, "variant": args.variant, "mode": args.mode,
                             "policy_seed": args.seed, "reset_seed": reset_seed,
                             "checkpoint_steps": step, "episodes": args.episodes,
                             "policy_updates": 0})
            for slot in sorted(episodes):
                traj = episodes[slot]
                group = h5.create_group(f"traj_{slot:03d}")
                for field in fields + ("mc_return_to_go",):
                    group.create_dataset(field, data=traj[field], compression="gzip", compression_opts=1)
                for attr in ("success", "episode_return", "final_angle_rad"):
                    group.attrs[attr] = traj[attr]
        temp.replace(output)
        successes = sum(int(traj["success"]) for traj in episodes.values())
        print(f"COLLECTED {name}: {len(episodes)} episodes, {successes} successes, "
              f"checkpoint={step}, path={output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
