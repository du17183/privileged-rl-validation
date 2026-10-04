"""Unselected expert validation and separately selected demonstration collection.

Only existing physical randomization and the closed-loop Cartesian/IK planner
are used. Old datasets, tasks, rewards and controllers are never modified.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['validate', 'collect'], required=True)
p.add_argument('--episodes', type=int, default=256)
p.add_argument('--num-envs', type=int, default=16)
p.add_argument('--max-attempts', type=int, default=600)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--planner', choices=['legacy','staged','staged_free'], default='legacy')
p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(headless=True).app

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from randomized_env.door_randomization import create
from door_env.door import robot_observation, SUCCESS_ANGLE_RAD
from door_env.isaac_env import transition_after_step
from door_dataset.planner import DoorWaypointPlanner
from experiments.phase13_random_expert_bc.planner import RandomDoorPlanner
from experiments.phase13_random_expert_bc.state import pack, next_environment, FIELDS


def wilson(k, n):
    z = 1.959963984540054
    x = k / n
    den = 1 + z*z/n
    middle = (x + z*z/(2*n)) / den
    delta = z*np.sqrt(x*(1-x)/n + z*z/(4*n*n)) / den
    return [float(middle-delta), float(middle+delta)]


def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists() or (out/'attempts.csv').exists():
        raise FileExistsError('Use a new output namespace; existing evidence is immutable')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    env = create(args.num_envs, args.device or 'cuda:0', args.seed, level=2)
    planner_class = DoorWaypointPlanner if args.planner == 'legacy' else RandomDoorPlanner
    planner = planner_class(env.num_envs, env.device, env.step_dt)
    if args.planner == 'staged_free': planner.release_orientation = True
    started = time.monotonic()
    buffers = [defaultdict(list) for _ in range(env.num_envs)]
    episode_counts = np.zeros(env.num_envs, dtype=int)
    quotas = np.full(env.num_envs, args.episodes//env.num_envs)
    quotas[:args.episodes % env.num_envs] += 1
    rows = []
    saved = successes = ticks = 0
    max_angle = torch.zeros(env.num_envs, device=env.device)
    contact_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    h5 = h5py.File(out/'trajectories.h5', 'x') if args.mode == 'collect' else None
    try:
        if h5 is not None:
            h5.attrs['environment_fields'] = json.dumps(FIELDS)
            h5.attrs['environment_dim'] = 13
            h5.attrs['robot_dim'] = 26
            h5.attrs['seed'] = args.seed
            h5.attrs['distribution'] = 'angle U(0,5 deg); rigid cabinet XYZ U(-.01,.01 m); friction unchanged'
            h5.attrs['expert'] = args.planner+'; measured handle/hinge Cartesian path; existing differential IK; physical contact'
            h5.attrs['action_noise_std'] = 0.
        with (out/'attempts.csv').open('x', newline='') as handle:
            columns = ['attempt','clone','episode','success','length','return','max_angle_rad','final_angle_rad',
                       'contact_seen','final_phase','angle0_rad','offset_x','offset_y','offset_z','friction_scale']
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            while True:
                robot = robot_observation(env).clone()
                gt = env.get_tool_state().clone()
                context = pack(env.get_environment_state()).clone()
                parameters = env.parameters.clone()
                phases = planner.state.clone()
                max_angle = torch.maximum(max_angle, gt[:, 0])
                contact_seen |= (gt[:, 9:11] > .5).all(-1)
                action = planner.action(env)
                _, reward, term, trunc, _ = env.step(action)
                next_robot, next_gt, done = transition_after_step(env, term, trunc)
                next_context = next_environment(env, done)
                max_angle = torch.maximum(max_angle, next_gt[:, 0])
                contact_seen |= (next_gt[:, 9:11] > .5).all(-1)
                batch = dict(observation=robot, environment_state=context, state=gt,
                             action=action, reward=reward.reshape(-1,1), next_observation=next_robot,
                             next_environment_state=next_context, next_state=next_gt,
                             done=done[:, None], terminated=term[:, None], truncated=trunc[:, None],
                             reset_parameters=parameters, planner_phase=phases[:, None])
                arrays = {key: value.cpu().numpy() for key, value in batch.items()}
                for i in range(env.num_envs):
                    if args.mode == 'validate' and episode_counts[i] >= quotas[i]:
                        continue
                    for key, value in arrays.items():
                        buffers[i][key].append(value[i].copy())
                ticks += 1
                ids = done.nonzero().flatten().tolist()
                for i in ids:
                    if args.mode == 'validate' and episode_counts[i] >= quotas[i]:
                        continue
                    if args.mode == 'collect' and (saved >= args.episodes or len(rows) >= args.max_attempts):
                        continue
                    data = {key: np.asarray(value) for key, value in buffers[i].items()}
                    ok = bool(next_gt[i, 0] > SUCCESS_ANGLE_RAD)
                    pars = data['reset_parameters'][0]
                    row = dict(attempt=len(rows),clone=i,episode=int(episode_counts[i]),success=int(ok),
                               length=len(data['action']),return_=float(data['reward'].sum()),
                               max_angle_rad=float(max_angle[i]),final_angle_rad=float(next_gt[i,0]),
                               contact_seen=int(contact_seen[i]),final_phase=int(planner.state[i]),
                               **dict(zip(columns[-5:], [float(x) for x in pars])))
                    row['return'] = row.pop('return_')
                    writer.writerow(row); rows.append(row); successes += int(ok)
                    episode_counts[i] += 1
                    if h5 is not None and ok:
                        group = h5.create_group(f'traj_{saved:05d}')
                        for key, value in data.items():
                            group.create_dataset(key, data=value, compression='gzip', compression_opts=1)
                        group['robot_state'] = group['observation']
                        group.create_dataset('success', data=np.full((len(data['action']),1), True))
                        for key, value in row.items(): group.attrs[key] = value
                        saved += 1
                    buffers[i] = defaultdict(list)
                if ids:
                    planner.reset(ids)
                    max_angle[ids] = 0
                    contact_seen[ids] = False
                    handle.flush()
                    if h5 is not None: h5.flush()
                if ticks % 100 == 0 or ids:
                    elapsed = time.monotonic()-started
                    print(json.dumps(dict(ticks=ticks,interactions=ticks*env.num_envs,
                                          completed=len(rows),successes=successes,saved=saved,
                                          phase_counts=torch.bincount(planner.state,minlength=6).cpu().tolist(),
                                          elapsed_s=round(elapsed,1),steps_per_s=round(ticks*env.num_envs/elapsed,1))), flush=True)
                if args.mode == 'validate' and np.all(episode_counts >= quotas): break
                if args.mode == 'collect' and (saved >= args.episodes or len(rows) >= args.max_attempts): break
                if ticks > 600*(args.max_attempts//env.num_envs+3): raise RuntimeError('Exceeded physical episode bound')
        summary = dict(mode=args.mode,seed=args.seed,episodes=len(rows),successes=successes,
                       success_rate=successes/len(rows),success_95_wilson=wilson(successes,len(rows)),
                       saved_trajectories=saved,vector_steps=ticks,simulator_interactions=ticks*env.num_envs,
                       elapsed_s=time.monotonic()-started,physical_reset_checks=env.physical_reset_checks,
                       num_envs=env.num_envs,distribution=dict(angle_deg=[0,5],xyz_m=[-.01,.01],friction_scale=1),
                       planner=args.planner,
                       mean_length=float(np.mean([r['length'] for r in rows])))
        if h5 is not None:
            for key in ['saved_trajectories','simulator_interactions','episodes','success_rate']: h5.attrs[key]=summary[key]
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('COMPLETE '+json.dumps(summary),flush=True)
        if args.mode == 'collect' and saved < args.episodes: raise RuntimeError('Demonstration quota not met')
    finally:
        if h5 is not None: h5.close()
        env.close()


if __name__ == '__main__':
    try: main()
    finally: app.close()
