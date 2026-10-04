"""Common heldout Door states for fair frozen-encoder probing.

All Phase 3 encoders are tested on the same baseline-policy trajectories.
These states never enter Phase 3 replay or gradient updates.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
from pathlib import Path
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from algorithms.asymmetric_sac import GaussianActor
from door_env.door import make_cfg, robot_observation
from door_env.isaac_env import create_env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'door_privileged_ablation' / 'representation_probe_states.npz'


@torch.no_grad()
def main():
    with (ROOT / 'results/door/eval_A_seed0.csv').open(newline='', encoding='utf-8') as stream:
        row = max(csv.DictReader(stream), key=lambda r: float(r['success_rate']))
    step = int(row['env_steps'])
    path = ROOT / 'checkpoints/door/A_seed0' / f'step_{step}.pt'
    cfg = make_cfg(32, device=args.device or 'cuda:0', seed=190000)
    env = create_env(cfg)
    try:
        actor = GaussianActor(26,7).to(env.device)
        actor.load_state_dict(torch.load(path, map_location=env.device, weights_only=False)['actor'])
        actor.eval()
        env.reset(seed=190000)
        counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        robots, states, ids = [], [], []
        for iteration in range(1806):
            robot = robot_observation(env)
            if iteration % 4 == 0:
                active = counts < 2
                robots.append(robot[active].cpu().numpy())
                states.append(env.get_tool_state()[active].cpu().numpy())
                ids.append(torch.arange(env.num_envs, device=env.device)[active].cpu().numpy())
            action, _ = actor(robot, deterministic=True)
            _, _, terminated, truncated, _ = env.step(action)
            counts += (terminated | truncated).long()
            if bool(torch.all(counts >= 2)): break
        if not bool(torch.all(counts >= 2)):
            raise RuntimeError(f'Only {int(counts.sum())}/64 heldout episodes')
        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT, robot=np.concatenate(robots), gt=np.concatenate(states),
                            env_id=np.concatenate(ids), source_checkpoint=str(path),
                            reset_seed=190000)
        print(f'Saved {sum(map(len,robots))} fixed heldout states to {OUT}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
