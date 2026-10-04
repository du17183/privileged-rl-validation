"""Independent deterministic/noisy and small initial-state-shift Door evaluation.

The fixture is changed only in this evaluator's configuration; no training
environment, reward, expert data, robot or baseline artifacts are modified.
"""

import argparse
import csv
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=("best", "final"), required=True)
parser.add_argument("--noise", type=float, default=0.0,
                    help="Additive pre-tanh Gaussian std; zero is deterministic")
parser.add_argument("--door-angle-deg", type=float, default=0.0)
parser.add_argument("--cabinet-dx", type=float, default=0.0)
parser.add_argument("--cabinet-dy", type=float, default=0.0)
parser.add_argument("--rounds", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.door_angle_deg < 0 or args.door_angle_deg > 90:
    parser.error("door_right_joint supports 0..90 degrees; closed nominal is the lower limit")
app = AppLauncher(headless=True).app

import numpy as np
import torch
import isaaclab_tasks  # noqa: F401

from auxiliary_learning.gt_prediction import EncodedGaussianActor
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation, door_angle, handle_pose
from door_env.isaac_env import create_env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase7_stable_policy" / "robustness"


def checkpoint_path():
    if args.variant in ("E0", "L000"):
        old = "B3" if args.variant == "E0" else "B1"
        run = f"P6{old}_seed{args.seed}"
        base = ROOT / "results" / "phase6_stable_online_rl"
        with (base / f"eval_{run}.csv").open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if int(r["env_steps"]) <= 300000]
        row = max(rows, key=lambda r: float(r["success_rate"])) if args.mode == "best" else rows[-1]
        step = int(row["env_steps"])
        return ROOT / "checkpoints" / "phase6_stable_online_rl" / run / f"step_{step}.pt", step
    run = f"P7{args.variant}_seed{args.seed}"
    base = ROOT / "results" / "phase7_stable_policy"
    with (base / f"eval_{run}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = max(rows, key=lambda r: float(r["success_rate"])) if args.mode == "best" else rows[-1]
    step = int(row["env_steps"])
    return ROOT / "checkpoints" / "phase7_stable_policy" / run / f"step_{step}.pt", step


@torch.no_grad()
def main():
    if args.noise < 0 or args.rounds < 1:
        raise ValueError("noise must be nonnegative and rounds positive")
    path, selected_step = checkpoint_path()
    torch.manual_seed(90000 + args.seed)
    cfg = make_cfg(32, device=args.device or "cuda:0", seed=90000 + args.seed)
    cfg.scene.cabinet.init_state.joint_pos["door_right_joint"] = math.radians(args.door_angle_deg)
    cfg.scene.cabinet.init_state.pos = (0.9 + args.cabinet_dx, args.cabinet_dy, 0.4)
    env = create_env(cfg)
    try:
        actor = EncodedGaussianActor(26, 7).to(env.device)
        actor.load_state_dict(torch.load(path, map_location=env.device,
                                         weights_only=False)["actor"])
        actor.eval()
        env.reset(seed=90000 + args.seed)
        initial_angle = float(door_angle(env).mean())
        initial_handle = handle_pose(env)[0] - env.scene.env_origins
        initial_handle_xyz = initial_handle.mean(dim=0).tolist()
        initial_policy = actor.policy_head(actor.encoder(robot_observation(env)))
        reset_mean_policy_std = float(initial_policy[:, 7:].clamp(-5.0, 2.0).exp().mean())
        counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        returns = torch.zeros(env.num_envs, device=env.device)
        successes, rewards = [], []
        interactions = 0
        max_steps = int(env.cfg.episode_length_s / env.step_dt + 2) * (args.rounds + 1)
        for _ in range(max_steps):
            observation = robot_observation(env)
            if args.noise == 0:
                action, _ = actor(observation, deterministic=True)
            else:
                mean = actor.policy_head(actor.encoder(observation))[:, :7]
                action = torch.tanh(mean + args.noise * torch.randn_like(mean))
            _, reward, terminated, truncated, _ = env.step(action)
            interactions += env.num_envs
            returns += reward
            done = terminated | truncated
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if counts[i] < args.rounds:
                    successes.append(float(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD))
                    rewards.append(float(returns[i]))
                    counts[i] += 1
                returns[i] = 0
            if bool(torch.all(counts >= args.rounds)):
                break
        expected = env.num_envs * args.rounds
        if len(successes) != expected:
            raise RuntimeError(f"Collected {len(successes)}/{expected} episodes")
        OUT.mkdir(parents=True, exist_ok=True)
        label = (f"{args.variant}_seed{args.seed}_{args.mode}_n{args.noise:g}_"
                 f"a{args.door_angle_deg:+g}_x{args.cabinet_dx:+g}_y{args.cabinet_dy:+g}")
        output = OUT / f"{label}.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("variant", "seed", "mode", "checkpoint_step", "noise_pre_tanh_std",
                             "requested_door_angle_deg", "requested_cabinet_dx_m", "requested_cabinet_dy_m",
                             "observed_initial_door_angle_rad", "observed_initial_handle_x_m",
                             "observed_initial_handle_y_m", "observed_initial_handle_z_m",
                             "reset_mean_policy_std", "success_rate", "mean_return",
                             "episodes", "eval_env_steps"))
            writer.writerow((args.variant, args.seed, args.mode, selected_step, args.noise,
                             args.door_angle_deg, args.cabinet_dx, args.cabinet_dy,
                             initial_angle, *initial_handle_xyz, reset_mean_policy_std,
                             float(np.mean(successes)),
                             float(np.mean(rewards)), expected, interactions))
        print(f"ROBUSTNESS {label} success={np.mean(successes):.3f} initial_angle={initial_angle:.3f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        app.close()
