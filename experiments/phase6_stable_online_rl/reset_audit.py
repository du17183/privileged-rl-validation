"""Read-only audit of fixed Door reset diversity; does not step or edit task."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import json
from pathlib import Path

import torch
import isaaclab_tasks  # noqa: F401

from door_env.door import make_cfg, robot_observation, privileged_state
from door_env.isaac_env import create_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase6_stable_online_rl" / "reset_diversity.json"


def stats(value):
    return {"max_dimension_std": float(value.std(dim=0, unbiased=False).max()),
            "unique_rows_1e-5": int(torch.unique(torch.round(value * 1e5).long(), dim=0).shape[0])}


def main():
    env = create_env(make_cfg(32, device=args.device or "cuda:0", seed=50000))
    try:
        results = []
        for seed in (50000, 90000):
            env.reset(seed=seed)
            robot = robot_observation(env)
            gt = privileged_state(env)
            results.append({"seed": seed, "robot": stats(robot),
                            "tool_state": stats(gt),
                            "door_angle_min": float(gt[:, 0].min()),
                            "door_angle_max": float(gt[:, 0].max())})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"num_envs": 32, "reset_results": results}, indent=2))
        print(OUT.read_text())
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
