"""Inspect physical rollout failure modes of a saved SAC actor."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--output", default="results/checkpoint_diagnostic.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import json
from pathlib import Path

import torch

import isaaclab_tasks  # noqa: F401

from algorithms.asymmetric_sac import AsymmetricSAC
from envs.drawer import SUCCESS_DISTANCE_M, drawer_position, make_cfg, robot_observation
from envs.isaac_env import create_env


def main():
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=901)
    env = create_env(cfg)
    try:
        robot_dim = robot_observation(env).shape[-1]
        gt_dim = env.get_tool_state().shape[-1]
        ckpt = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        agent = AsymmetricSAC(robot_dim, gt_dim, env.action_space.shape[-1], ckpt["variant"], device=env.device)
        agent.actor.load_state_dict(ckpt["actor"])
        maxima = torch.zeros(env.num_envs, device=env.device)
        contact_steps = torch.zeros(env.num_envs, device=env.device)
        min_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
        done_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        successes = torch.zeros(env.num_envs, device=env.device)
        trace = []
        simulator_interactions = 0
        horizon = int(cfg.episode_length_s / env.step_dt) + 2
        with torch.no_grad():
            for step in range(horizon):
                robot = robot_observation(env)
                gt = env.get_tool_state()
                active = ~done_once
                maxima[active] = torch.maximum(maxima[active], drawer_position(env).squeeze(-1)[active])
                contact_steps[active] += (gt[active, -2:] > 0.5).all(dim=-1).float()
                ee = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
                handle = env.scene["cabinet_frame"].data.target_pos_w[:, 0, :]
                distance = torch.linalg.vector_norm(ee - handle, dim=-1)
                min_distance[active] = torch.minimum(min_distance[active], distance[active])
                action = agent.act(robot, gt if agent.variant == "C" else None, deterministic=True)
                if step % 20 == 0:
                    trace.append({
                        "step": step,
                        "tcp_handle_distance_m": float(distance[0]),
                        "drawer_m": float(drawer_position(env)[0, 0]),
                        "gripper_action": float(action[0, -1]),
                        "arm_action_norm": float(torch.linalg.vector_norm(action[0, :6])),
                        "finger_joint_position_m": float(robot[0, 7]),
                    })
                _, _, terminated, truncated, _ = env.step(action)
                simulator_interactions += env.num_envs
                newly_done = (terminated | truncated) & active
                if newly_done.any():
                    terminal = env.final_drawer_position[:, 0]
                    maxima[newly_done] = torch.maximum(maxima[newly_done], terminal[newly_done])
                    successes[newly_done] = (terminal[newly_done] > SUCCESS_DISTANCE_M).float()
                    done_once[newly_done] = True
                if bool(done_once.all()):
                    break
        report = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "variant": agent.variant,
            "env_steps": ckpt["env_steps"],
            "episodes": env.num_envs,
            "simulator_interactions": simulator_interactions,
            "success_rate": float(successes.mean()),
            "grasp_rate": float((contact_steps > 0).float().mean()),
            "contact_assisted_success_rate": float(((contact_steps > 0) & (successes > 0)).float().mean()),
            "max_drawer_m_mean": float(maxima.mean()),
            "max_drawer_m_per_env": maxima.tolist(),
            "min_tcp_handle_distance_m_mean": float(min_distance.mean()),
            "both_fingers_contact_steps_mean": float(contact_steps.mean()),
            "first_environment_trace_every_20_steps": trace,
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        detail = traceback.format_exc()
        Path("logs/diagnose_error.txt").write_text(detail, encoding="utf-8")
        print(detail, flush=True)
        raise
    finally:
        app.close()
