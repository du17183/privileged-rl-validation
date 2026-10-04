"""Independent best/final Door checkpoint replay on 64 heldout episodes."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=('E0','E25','E50','E100','E200','E100RB','E100R1','E100M','Q0','QGT','QV'), required=True)
parser.add_argument('--seed', type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
from pathlib import Path
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from auxiliary_learning.gt_prediction import EncodedGaussianActor
from auxiliary_learning.multitask_encoder import MultiTaskActor
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'stable_privileged_rl'


def checkpoint(mode):
    path = OUT / f'eval_{args.variant}_seed{args.seed}.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    row = max(rows, key=lambda r: float(r['success_rate'])) if mode == 'best' else rows[-1]
    step = int(row['env_steps'])
    return (ROOT / 'checkpoints' / 'stable_privileged_rl' /
            f'{args.variant}_seed{args.seed}' / f'step_{step}.pt'), step, float(row['success_rate'])


@torch.no_grad()
def one_mode(env, mode):
    path, step, selection_rate = checkpoint(mode)
    actor = (MultiTaskActor(26, 7) if args.variant == 'E100M'
             else EncodedGaussianActor(26, 7)).to(env.device)
    actor.load_state_dict(torch.load(path, map_location=env.device, weights_only=False)['actor'])
    actor.eval()
    env.reset(seed=90000 + args.seed)
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success, reward_list, contact_list, final_angle_list = [], [], [], []
    angle_errors, contact_correct = [], []
    interactions = 0
    for iteration in range(1806):
        robot = robot_observation(env)
        if iteration % 10 == 0:
            gt = env.get_tool_state()
            prediction = actor.predict_gt(robot)
            angle_errors.append(float((prediction[:,0] - gt[:,0]).abs().mean()))
            contact_correct.append(float(((prediction[:,1:].sigmoid() > .5) ==
                                          (gt[:,9:11] > .5)).float().mean()))
        action, _ = actor(robot, deterministic=True)
        _, reward, terminated, truncated, _ = env.step(action)
        interactions += env.num_envs
        returns += reward
        done = terminated | truncated
        gt = env.get_tool_state()
        contact = (gt[:, -2:] > .5).all(dim=-1)
        if done.any():
            contact[done] = (env.final_privileged[done, -2:] > .5).all(dim=-1)
        contact_seen |= contact
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            if counts[i] < 2:
                success.append(float(env.final_door_angle[i,0] > SUCCESS_ANGLE_RAD))
                final_angle_list.append(float(env.final_door_angle[i,0]))
                reward_list.append(float(returns[i]))
                contact_list.append(float(contact_seen[i]))
                counts[i] += 1
            returns[i] = 0
            contact_seen[i] = False
        if bool(torch.all(counts >= 2)):
            break
    if len(success) != 64:
        raise RuntimeError(f'{args.variant} seed={args.seed} {mode}: {len(success)}/64 episodes')
    result = (args.variant, args.seed, mode, step, selection_rate,
              float(np.mean(success)), float(np.mean(reward_list)),
              float(np.mean(contact_list)), float(np.mean(final_angle_list)),
              len(success), interactions,
              float(np.mean(angle_errors)) if angle_errors else '',
              float(np.mean(contact_correct)) if contact_correct else '')
    out = OUT / f'heldout_{args.variant}_seed{args.seed}_{mode}.csv'
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('variant','seed','mode','selected_steps','selection_success_rate',
                         'heldout_success_rate','heldout_mean_return','heldout_contact_rate',
                         'heldout_final_angle_rad',
                         'episodes','eval_env_steps','angle_prediction_mae_rad',
                         'contact_prediction_accuracy'))
        writer.writerow(result)
    print('HELDOUT',result,flush=True)


def main():
    cfg = make_cfg(32, device=args.device or 'cuda:0', seed=90000 + args.seed)
    env = create_env(cfg)
    try:
        for mode in ('best','final'):
            one_mode(env,mode)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
