"""Independent fixed-reset verification of Phase 6 BC, best, and final actors."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A0", "A1", "A2", "B1", "B2", "B3"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=("initial", "best", "final"), required=True)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--action-std-cap", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
from pathlib import Path

import numpy as np
import torch
import isaaclab_tasks  # noqa: F401

from auxiliary_learning.gt_prediction import EncodedGaussianActor
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase6_stable_online_rl"
CHECKPOINTS = ROOT / "checkpoints" / "phase6_stable_online_rl"


def select_checkpoint():
    with (OUT / f"eval_P6{args.variant}_seed{args.seed}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.mode == "initial":
        row = rows[0]
    elif args.mode == "best":
        row = max(rows, key=lambda r: float(r["success_rate"]))
    else:
        row = rows[-1]
    step = int(row["env_steps"])
    path = CHECKPOINTS / f"P6{args.variant}_seed{args.seed}" / f"step_{step}.pt"
    return path, step, float(row["success_rate"])


@torch.no_grad()
def main():
    if args.action_std_cap is not None and (not args.stochastic or args.action_std_cap <= 0):
        raise ValueError("A positive action std cap requires --stochastic")
    path, step, selection_rate = select_checkpoint()
    cfg = make_cfg(32, device=args.device or "cuda:0", seed=90000 + args.seed)
    env = create_env(cfg)
    try:
        actor = EncodedGaussianActor(26, 7).to(env.device)
        actor.load_state_dict(torch.load(path, map_location=env.device,
                                         weights_only=False)["actor"])
        actor.eval()
        env.reset(seed=90000 + args.seed)
        reset_robot = robot_observation(env)
        reset_policy = actor.policy_head(actor.encoder(reset_robot))
        _, reset_log_std = reset_policy.split(7, dim=-1)
        reset_action_std = float(reset_log_std.clamp(-5.0, 2.0).exp().mean())
        counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        returns = torch.zeros(env.num_envs, device=env.device)
        successes, rewards, angles = [], [], []
        interactions = 0
        max_steps = int(env.cfg.episode_length_s / env.step_dt + 2) * 3
        for _ in range(max_steps):
            observation = robot_observation(env)
            if args.action_std_cap is None:
                action, _ = actor(observation, deterministic=not args.stochastic)
            else:
                mean, log_std = actor.policy_head(actor.encoder(observation)).split(7, dim=-1)
                std = log_std.clamp(-5.0, 2.0).exp().clamp(max=args.action_std_cap)
                action = torch.tanh(torch.distributions.Normal(mean, std).sample())
            _, reward, terminated, truncated, _ = env.step(action)
            interactions += env.num_envs
            returns += reward
            done = terminated | truncated
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if counts[i] < 2:
                    successes.append(float(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD))
                    angles.append(float(env.final_door_angle[i, 0]))
                    rewards.append(float(returns[i]))
                    counts[i] += 1
                returns[i] = 0
            if bool(torch.all(counts >= 2)):
                break
        if len(successes) != 64:
            raise RuntimeError(f"Collected only {len(successes)}/64 episodes")
        suffix = "_stochastic" if args.stochastic else ""
        if args.action_std_cap is not None:
            suffix += "_cap" + str(args.action_std_cap).replace(".", "p")
        output = OUT / f"heldout_P6{args.variant}_seed{args.seed}_{args.mode}{suffix}.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("variant", "seed", "mode", "stochastic", "selected_steps",
                             "selection_success_rate", "heldout_success_rate",
                             "heldout_mean_return", "heldout_final_angle_rad",
                             "episodes", "eval_env_steps", "reset_mean_action_std",
                             "action_std_cap"))
            writer.writerow((args.variant, args.seed, args.mode, int(args.stochastic),
                             step, selection_rate, float(np.mean(successes)),
                             float(np.mean(rewards)), float(np.mean(angles)),
                             len(successes), interactions, reset_action_std,
                             args.action_std_cap if args.action_std_cap is not None else ""))
        print(f"HELDOUT {args.variant} seed={args.seed} mode={args.mode} stochastic={args.stochastic} success={np.mean(successes):.3f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
