"""Collect successful physical Panda door demonstrations into HDF5."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--max-attempts", type=int, default=3000)
parser.add_argument("--seed", type=int, default=140)
parser.add_argument("--output", default="door_dataset/door_expert_1000.h5")
parser.add_argument("--arm-noise-std", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import json
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env, transition_after_step
from door_dataset.planner import DoorWaypointPlanner


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=args.seed)
    env = create_env(cfg)
    planner = DoorWaypointPlanner(env.num_envs, env.device, env.step_dt)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffers = [defaultdict(list) for _ in range(env.num_envs)]
    successes = attempts = vector_steps = 0
    lengths = []
    try:
        with h5py.File(path, "w") as h5:
            h5.attrs["env_id"] = "PandaSektionDoorRight-v0"
            h5.attrs["expert_type"] = "closed-loop Cartesian waypoints + differential IK + physical contact"
            h5.attrs["seed"] = args.seed
            h5.attrs["num_envs"] = env.num_envs
            h5.attrs["success_angle_rad"] = SUCCESS_ANGLE_RAD
            h5.attrs["arm_noise_std"] = args.arm_noise_std
            h5.attrs["state_fields"] = json.dumps([
                "door_angle", "door_angular_velocity", "handle_xyz", "handle_quat_wxyz",
                "left_contact", "right_contact"])
            h5.attrs["robot_fields"] = json.dumps([
                "joint_position_9", "joint_velocity_9", "tcp_xyz", "tcp_quat_wxyz", "episode_progress"])
            while successes < args.episodes and attempts < args.max_attempts:
                robot = robot_observation(env).clone()
                state = env.get_tool_state().clone()
                action = planner.action(env)
                if args.arm_noise_std:
                    action = action.clone()
                    action[:, :6] = (action[:, :6] + args.arm_noise_std * torch.randn_like(action[:, :6])).clamp(-1, 1)
                _, reward, terminated, truncated, _ = env.step(action)
                next_robot, next_state, done = transition_after_step(env, terminated, truncated)
                batch = {
                    "observation": robot.cpu().numpy(),
                    "state": state.cpu().numpy(),
                    "action": action.cpu().numpy(),
                    "reward": reward.reshape(-1, 1).cpu().numpy(),
                    "next_observation": next_robot.cpu().numpy(),
                    "next_state": next_state.cpu().numpy(),
                    "done": done.reshape(-1, 1).cpu().numpy(),
                    "terminated": terminated.reshape(-1, 1).cpu().numpy(),
                    "truncated": truncated.reshape(-1, 1).cpu().numpy(),
                }
                for i in range(env.num_envs):
                    for key, value in batch.items():
                        buffers[i][key].append(value[i].copy())
                if done.any():
                    ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
                    for i in ids:
                        attempts += 1
                        finished = bool(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD)
                        if finished and successes < args.episodes:
                            group = h5.create_group(f"traj_{successes:05d}")
                            for key, values in buffers[i].items():
                                group.create_dataset(key, data=np.asarray(values), compression="gzip", compression_opts=1)
                            group["robot_state"] = group["observation"]
                            group["privileged_state"] = group["state"]
                            group["next_privileged_state"] = group["next_state"]
                            group.attrs["success"] = True
                            group.attrs["length"] = len(buffers[i]["action"])
                            group.attrs["final_angle_rad"] = float(env.final_door_angle[i, 0])
                            lengths.append(len(buffers[i]["action"]))
                            successes += 1
                            if successes % 25 == 0:
                                h5.flush()
                                print(f"successes={successes} attempts={attempts} vector_steps={vector_steps} mean_len={np.mean(lengths):.1f}", flush=True)
                        buffers[i] = defaultdict(list)
                    planner.reset(ids)
                vector_steps += 1
            h5.attrs["successful_episodes"] = successes
            h5.attrs["attempts"] = attempts
            h5.attrs["vector_steps"] = vector_steps
            h5.attrs["simulator_interactions"] = vector_steps * env.num_envs
            h5.flush()
    finally:
        env.close()
    if successes < args.episodes:
        raise RuntimeError(f"Only {successes}/{args.episodes} successful demonstrations after {attempts} attempts")
    print(f"SAVED {successes} {path} mean_len={np.mean(lengths):.1f} attempts={attempts}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
