"""Runtime checks for the Panda drawer fixture and automatic reset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--repeats", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import json
import statistics
import time
from pathlib import Path

import torch

import isaaclab_tasks  # noqa: F401

from envs.drawer import drawer_index, drawer_position, drawer_velocity, make_cfg, privileged_state, robot_observation
from envs.isaac_env import create_env


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=41)
    env = create_env(cfg)
    times = []
    passed = 0
    try:
        robot_dim = robot_observation(env).shape[-1]
        gt_dim = privileged_state(env).shape[-1]
        action_dim = env.action_space.shape[-1]
        cabinet = env.scene["cabinet"]
        joint_id = drawer_index(env)
        home = env.scene["robot"].data.default_joint_pos
        for _ in range(args.repeats):
            # Perturb the drawer and robot so this exercises restoration, not only idempotence.
            cabinet_q = cabinet.data.joint_pos.clone()
            cabinet_v = cabinet.data.joint_vel.clone()
            cabinet_q[:, joint_id] = 0.15
            cabinet.write_joint_state_to_sim(cabinet_q, cabinet_v)
            robot = env.scene["robot"]
            robot_q = robot.data.joint_pos.clone()
            robot_v = robot.data.joint_vel.clone()
            robot_q[:, 0] += 0.05
            robot.write_joint_state_to_sim(robot_q, robot_v)
            torch.cuda.synchronize()
            start = time.perf_counter()
            env.reset()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            closed = bool(torch.all(drawer_position(env).abs() < 0.02))
            still = bool(torch.all(drawer_velocity(env).abs() < 0.05))
            homed = bool(torch.all((robot.data.joint_pos - home).abs() < 0.02))
            if closed and still and homed:
                passed += 1
        env.reset()
        expected_ee = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
        expected_handle = env.scene["cabinet_frame"].data.target_pos_w[:, 0, :].clone()
        cabinet_q = cabinet.data.joint_pos.clone()
        cabinet_v = cabinet.data.joint_vel.clone()
        cabinet_q[:, joint_id] = 0.35
        cabinet.write_joint_state_to_sim(cabinet_q, cabinet_v)
        _, _, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, action_dim), device=env.device))
        auto_reset_done = bool(torch.all(terminated | truncated))
        auto_reset_ee_error = float(torch.linalg.vector_norm(
            env.scene["ee_frame"].data.target_pos_w[:, 0, :] - expected_ee, dim=-1
        ).max())
        auto_reset_handle_error = float(torch.linalg.vector_norm(
            env.scene["cabinet_frame"].data.target_pos_w[:, 0, :] - expected_handle, dim=-1
        ).max())
        auto_reset_closed = bool(torch.all(drawer_position(env).abs() < 0.02))
        auto_reset_home = bool(torch.all((env.scene["robot"].data.joint_pos - home).abs() < 0.02))
        env.reset()
        timeout_reset_truncated = False
        timeout_step_count = 0
        for timeout_step_count in range(1, env.max_episode_length + 3):
            _, _, _, timeout_flags, _ = env.step(
                torch.zeros((env.num_envs, action_dim), device=env.device)
            )
            if bool(torch.all(timeout_flags)):
                timeout_reset_truncated = True
                break
        timeout_reset_closed = bool(torch.all(drawer_position(env).abs() < 0.02))
        timeout_reset_home = bool(torch.all((env.scene["robot"].data.joint_pos - home).abs() < 0.02))
        report = {
            "num_envs": env.num_envs,
            "robot_observation_dim": robot_dim,
            "privileged_state_dim": gt_dim,
            "action_dim": action_dim,
            "drawer_joint_name": cabinet.joint_names[joint_id],
            "drawer_joint_position_after_reset_m": drawer_position(env).squeeze(-1).tolist(),
            "contact_filter_shapes": {
                side: list(env.scene[side].data.force_matrix_w.shape)
                for side in ("left_contact", "right_contact")
            },
            "reset_attempts": args.repeats,
            "reset_successes": passed,
            "reset_success_rate": passed / args.repeats,
            "reset_wall_time_mean_s": statistics.mean(times),
            "reset_wall_time_p95_s": sorted(times)[max(0, int(0.95 * len(times)) - 1)],
            "automatic_reset_done_all": auto_reset_done,
            "automatic_reset_closed": auto_reset_closed,
            "automatic_reset_home": auto_reset_home,
            "automatic_reset_ee_frame_error_m": auto_reset_ee_error,
            "automatic_reset_handle_frame_error_m": auto_reset_handle_error,
            "timeout_reset_step_count": timeout_step_count,
            "timeout_reset_truncated_all": timeout_reset_truncated,
            "timeout_reset_closed": timeout_reset_closed,
            "timeout_reset_home": timeout_reset_home,
        }
        path = root / "results" / "drawer_preflight.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if passed != args.repeats:
            raise RuntimeError(f"Only {passed}/{args.repeats} resets restored Panda home and closed drawer")
        if not all((auto_reset_done, auto_reset_closed, auto_reset_home)) or max(auto_reset_ee_error, auto_reset_handle_error) > 0.02:
            raise RuntimeError("Automatic reset did not restore coherent Panda and drawer frame state")
        if not all((timeout_reset_truncated, timeout_reset_closed, timeout_reset_home)):
            raise RuntimeError("Timeout failure did not automatically restore Panda home and closed drawer")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        detail = traceback.format_exc()
        error_path = Path(__file__).resolve().parents[1] / "logs/validate_error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(detail, encoding="utf-8")
        print(detail, flush=True)
        raise
    finally:
        app.close()
