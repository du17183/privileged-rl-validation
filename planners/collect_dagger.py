"""Label corrective states with the privileged planner for BC aggregation.

These rollouts are additional simulator interactions and are reported separately
from the 500-1000 successful expert demonstrations and online SAC steps.
The rollout actor sees robot signals only; GT is used by the training oracle.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--actor-checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--expert-prob", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

from pathlib import Path

import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

from algorithms.asymmetric_sac import AsymmetricSAC
from envs.drawer import SUCCESS_DISTANCE_M, make_cfg, robot_observation
from envs.isaac_env import create_env
from planners.scripted_drawer import DrawerWaypointPlanner


def main():
    if args.steps <= 0 or not 0 <= args.expert_prob <= 1:
        raise ValueError("steps must be positive and expert-prob must be in [0,1]")
    torch.manual_seed(args.seed)
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=args.seed)
    env = create_env(cfg)
    planner = DrawerWaypointPlanner(env.num_envs, env.device, env.step_dt)
    checkpoint = torch.load(args.actor_checkpoint, map_location=env.device, weights_only=False)
    if checkpoint["variant"] != "A":
        raise ValueError("DAgger rollout actor must be robot-only variant A")
    actor = AsymmetricSAC(
        robot_observation(env).shape[-1], env.get_tool_state().shape[-1],
        env.action_space.shape[-1], "A", device=env.device,
    )
    actor.actor.load_state_dict(checkpoint["actor"])
    actor.actor.eval()
    arrays = {key: [] for key in ("observation", "state", "action", "executed_action", "done")}
    successes = attempts = 0
    try:
        with torch.no_grad():
            while len(arrays["observation"]) * env.num_envs < args.steps:
                robot = robot_observation(env).clone()
                gt = env.get_tool_state().clone()
                oracle = planner.action(env)
                policy = actor.act(robot, deterministic=True)
                choose_expert = torch.rand(env.num_envs, device=env.device) < args.expert_prob
                executed = torch.where(choose_expert[:, None], oracle, policy)
                _, _, terminated, truncated, _ = env.step(executed)
                done = terminated | truncated
                for key, tensor in (
                    ("observation", robot), ("state", gt), ("action", oracle),
                    ("executed_action", executed), ("done", done[:, None]),
                ):
                    arrays[key].append(tensor.cpu().numpy())
                if done.any():
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    attempts += len(ids)
                    successes += int((env.final_drawer_position[ids, 0] > SUCCESS_DISTANCE_M).sum())
                    planner.reset(ids)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as h5:
            for key, values in arrays.items():
                h5.create_dataset(key, data=np.concatenate(values, axis=0), compression="gzip", compression_opts=1)
            h5.attrs["actor_checkpoint"] = str(Path(args.actor_checkpoint).resolve())
            h5.attrs["expert_action_probability"] = args.expert_prob
            h5.attrs["simulator_interactions"] = len(arrays["observation"]) * env.num_envs
            h5.attrs["completed_episodes"] = attempts
            h5.attrs["successes"] = successes
            h5.attrs["seed"] = args.seed
        print(f"Saved {output}: interactions={len(arrays['observation']) * env.num_envs} "
              f"completed={attempts} success={successes}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
