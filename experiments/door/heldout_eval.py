"""Independent 64-episode replay of selected best or final Door checkpoints."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A", "B", "C", "D"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=("best", "final"), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
from pathlib import Path
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from algorithms.asymmetric_sac import AsymmetricSAC
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env

ROOT = Path(__file__).resolve().parents[2]


def selected_checkpoint():
    path = ROOT / "results/door" / f"eval_{args.variant}_seed{args.seed}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if args.mode == "best":
        row = max(rows, key=lambda r: float(r["success_rate"]))
    else:
        row = rows[-1]
    step = int(row["env_steps"])
    checkpoint = ROOT / "checkpoints/door" / f"{args.variant}_seed{args.seed}" / f"step_{step}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint, step, float(row["success_rate"])


@torch.no_grad()
def run():
    checkpoint, step, selection_rate = selected_checkpoint()
    cfg = make_cfg(32, device=args.device or "cuda:0", seed=90000 + args.seed)
    env = create_env(cfg)
    try:
        agent = AsymmetricSAC(26, 11, 7, "B" if args.variant == "D" else args.variant, device=env.device)
        agent.actor.load_state_dict(torch.load(checkpoint, map_location=env.device, weights_only=False)["actor"])
        env.reset(seed=90000 + args.seed)
        counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        returns = torch.zeros(env.num_envs, device=env.device)
        contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        success, reward_list, contact_list = [], [], []
        interactions = 0
        for _ in range(1806):
            robot = robot_observation(env)
            action = agent.act(robot, env.get_tool_state() if args.variant == "C" else None,
                               deterministic=True)
            _, reward, terminated, truncated, _ = env.step(action)
            interactions += env.num_envs
            returns += reward
            done = terminated | truncated
            gt = env.get_tool_state()
            contact = (gt[:, -2:] > 0.5).all(dim=-1)
            if done.any():
                contact[done] = (env.final_privileged[done, -2:] > 0.5).all(dim=-1)
            contact_seen |= contact
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if counts[i] < 2:
                    success.append(float(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD))
                    reward_list.append(float(returns[i]))
                    contact_list.append(float(contact_seen[i]))
                    counts[i] += 1
                returns[i] = 0
                contact_seen[i] = False
            if bool(torch.all(counts >= 2)):
                break
        if len(success) != 64:
            raise RuntimeError(f"Only {len(success)}/64 held-out episodes")
        result = (args.variant, args.seed, args.mode, step, selection_rate,
                  float(np.mean(success)), float(np.mean(reward_list)),
                  float(np.mean(contact_list)), len(success), interactions)
        out = ROOT / "results/door" / f"heldout_{args.variant}_seed{args.seed}_{args.mode}.csv"
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("variant", "seed", "mode", "selected_steps", "selection_success_rate",
                             "heldout_success_rate", "heldout_mean_return", "heldout_contact_rate",
                             "episodes", "eval_env_steps"))
            writer.writerow(result)
        print("HELDOUT",result,flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        run()
    finally:
        app.close()
