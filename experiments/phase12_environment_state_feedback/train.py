"""Bounded matched continuation; measured Actor/Critic and uniform 50/50 replay."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--arm', required=True, choices=('A', 'B', 'C', 'D'))
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--steps', type=int, default=300000)
parser.add_argument('--pilot', action='store_true')
parser.add_argument('--strength', choices=('strong','medium','weak','none'), default='strong')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import copy
import csv
import hashlib
import json
import math
import os
import time
import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks
from algorithms.replay import ReplayBuffer, mixed_sample
from randomized_env.door_randomization import create
from types import SimpleNamespace
from safe_online.safe_update import SafeUpdate
from safe_online.kl_constraint import anchor_kl
from progress_rl.progress_monitor import EpisodeMonitor

from environment_feedback.normalization import DIMS
from experiments.phase12_environment_state_feedback.agent import MeasuredSAC
from experiments.phase12_environment_state_feedback.data import MeasuredExpert, live_observation, transition
from experiments.phase12_environment_state_feedback.eval_client import EvaluationClient
from experiments.phase12_environment_state_feedback.protocol import ROOT, OUT, LOG, CKPT, prepared, STRENGTHS


def append(path, row):
    first = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if first: writer.writeheader()
        writer.writerow(row)


def main():
    if args.steps % 32 or (not args.pilot and args.steps != 300000): raise ValueError('Budget changed')
    protocol = prepared()
    run = f'P12{args.arm}_{args.strength}_seed{args.seed}'+('_pilot' if args.pilot else '')
    folder = CKPT/run;folder.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    source_path = ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{args.seed}'/'step_300000.pt'
    anchor_path = ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt'
    hashes = dict(source=hashlib.sha256(source_path.read_bytes()).hexdigest(), anchor=hashlib.sha256(anchor_path.read_bytes()).hexdigest())
    for key in hashes:
        if hashes[key] != protocol['source_hashes'][f'{key}_{args.seed}']: raise RuntimeError('Frozen initialization changed')
    source = torch.load(source_path, map_location=args.device or 'cuda:0', weights_only=False)
    anchor = torch.load(anchor_path, map_location=args.device or 'cuda:0', weights_only=False)
    curriculum = SimpleNamespace(level=2, enabled=False, history=[])  # Full train/test distribution from first reset
    env = create(32, args.device or 'cuda:0', args.seed, curriculum.level)
    agent = MeasuredSAC(source, anchor, args.arm, env.device, STRENGTHS[args.strength])
    collector = copy.deepcopy(agent.actor).eval();collector.requires_grad_(False)
    expert = MeasuredExpert(ROOT/'door_dataset/door_expert_1000.h5', env.device, env.step_dt, args.arm, protocol['handle_center'])
    online = ReplayBuffer(args.steps+32, DIMS[args.arm], 11, 7, env.device)
    torch.manual_seed(300000+args.seed)
    reference = expert.sample(2048)['robot'].clone()
    monitor = EpisodeMonitor(32, env.device)
    lengths = torch.zeros(32, device=env.device, dtype=torch.long)
    returns = torch.zeros(32, device=env.device)
    moving_ticks, contacts = torch.zeros_like(returns), torch.zeros_like(returns)
    initial_params = env.parameters.clone()
    initial_handle = env.get_environment_state()['handle_position'].clone()
    initial_level = env.episode_level.clone()
    starts = torch.arange(32, device=env.device)
    trace_records = []
    steps = updates = successes = episodes = eval_steps = version = 0
    guard = SafeUpdate()
    best_score = None
    best_step = 0
    losses = []
    next_guard = 10000
    start_time = time.perf_counter()
    client = None
    writer = SummaryWriter(str(LOG/run))

    def save(path, validation):
        state = agent.checkpoint()
        state.update(env_steps=steps, seed=args.seed, initialization_sha256=hashes,
            phase12_config=protocol, anchor_strength=args.strength, curriculum_level=curriculum.level, validation=validation)
        torch.save(state, path)

    def call_pairs(actor, conditions, seed, rounds=1):
        nonlocal eval_steps
        metrics, raw = client.pairs(actor, conditions, seed, rounds)
        eval_steps += sum(r['metrics']['eval_env_steps'] for r in raw.values())
        return metrics

    def formal():
        nonlocal best_score, best_step
        tests = [dict(condition=c, mode='policy', seed=71000+args.seed, rounds=2) for c in ('nominal', 'level2')]
        raw = client.tests(agent.actor, tests)
        nonlocal eval_steps
        eval_steps += sum(r['metrics']['eval_env_steps'] for r in raw.values())
        results = {c: raw[f'{c}:policy:{71000+args.seed}']['metrics'] for c in ('nominal', 'level2')}
        for condition, metrics in results.items():
            append(OUT/f'eval_{run}_{condition}.csv', dict(env_steps=steps,
                **{k: v for k, v in metrics.items() if isinstance(v, (float, int, str))},
                online_successes=successes, online_episodes=episodes, curriculum_level=curriculum.level,
                rejections=guard.rejections, cumulative_eval_steps=eval_steps, wall_time_s=time.perf_counter()-start_time))
            for key, value in metrics.items():
                if isinstance(value, (float, int)): writer.add_scalar(f'eval/{condition}/{key}', value, steps)
        # Randomized success is the research objective; nominal preservation is
        # evaluated independently and protected by the same candidate guard.
        score = (results['level2']['success'], results['nominal']['success'], results['level2']['progress'])
        save(folder/f'step_{steps}.pt', results)
        if best_score is None or score > best_score:
            best_score, best_step = score, steps;save(folder/'best.pt', results)
        return results

    try:
        client = EvaluationClient(ROOT, args.seed, args.arm, run)
        conditions = ['nominal', 'level2']
        incumbent = call_pairs(collector, conditions, 70000+args.seed)
        formal()
        env.reset(seed=args.seed)
        initial_params.copy_(env.parameters);initial_handle.copy_(env.get_environment_state()['handle_position']);initial_level.copy_(env.episode_level)
        torch.manual_seed(400000+args.seed)
        guard.begin(agent)
        while steps < args.steps:
            observation = live_observation(env, args.arm, protocol['handle_center']).clone()
            gt = env.get_tool_state().clone()
            with torch.no_grad(): action = collector(observation)[0]
            _, reward, terminated, truncated, _ = env.step(action)
            next_obs, next_gt, done = transition(env, terminated, truncated, args.arm, protocol['handle_center'])
            steps += 32;lengths += 1;returns += reward
            moving = next_gt[:, 0] > initial_params[:, 0]+.02
            moving_ticks += moving;contacts += moving & (next_gt[:, 9:11]>.5).all(-1)
            monitor.update(gt[:, :1], next_gt[:, :1], (next_gt[:, 9:11]>.5).all(-1, keepdim=True))
            for i in done.nonzero().squeeze(-1).tolist():
                row = monitor.finish(i, next_gt[i, 0], float(env.final_target[i, 0]), lengths[i], float(env.final_start[i, 0]))
                episodes += 1;successes += int(row['success'])
                contact_stability = float(contacts[i]/moving_ticks[i].clamp_min(1))
                p = initial_params[i]
                row.update(env_steps=steps, env_index=i, policy_version=version, return_value=float(returns[i]),
                    initial_angle_deg=math.degrees(float(p[0])), offset_x_m=float(p[1]), offset_y_m=float(p[2]), offset_z_m=float(p[3]),
                    offset_linf_cm=100*float(p[1:4].abs().max()), friction_scale=float(p[4]),
                    initial_handle_x_m=float(initial_handle[i, 0]), initial_handle_y_m=float(initial_handle[i, 1]), initial_handle_z_m=float(initial_handle[i, 2]),
                    final_handle_x_m=float(env.final_environment['handle_position'][i, 0]),
                    final_handle_y_m=float(env.final_environment['handle_position'][i, 1]),
                    final_handle_z_m=float(env.final_environment['handle_position'][i, 2]),
                    curriculum_level=int(initial_level[i]), contact_stability=contact_stability)
                append(OUT/f'episodes_{run}.csv', row)
                trace_records.append(dict(start=int(starts[i]), length=int(lengths[i]), stride=32, **row))
                starts[i] = steps+i
                initial_params[i] = env.parameters[i];initial_handle[i] = env.get_environment_state()['handle_position'][i];initial_level[i] = env.episode_level[i]
                contacts[i] = moving_ticks[i] = returns[i] = 0;lengths[i] = 0
            online.add(dict(robot=observation, privileged=gt, action=action, reward=reward[:, None],
                next_robot=next_obs, next_privileged=next_gt, done=done.float()[:, None]))
            for _ in range(4):
                batch = mixed_sample(online, expert, 256, .5)
                losses.append(agent.online_update(batch, expert.sample(256), 10.))
                updates += 1
            if steps >= next_guard or steps == args.steps:
                candidate = call_pairs(agent.actor, conditions, 70000+args.seed)
                reasons = [f'{condition}/{r}' for condition in conditions for r in guard.compare(incumbent[condition], candidate[condition])]
                save(folder/f'raw_step_{steps}.pt', candidate)
                if reasons:
                    agent.load_state(guard.before);guard.rejections += 1;guard.rollback_events += 1;accepted = False
                else:
                    guard.acceptances += 1;accepted = True
                    collector.load_state_dict(agent.actor.state_dict());collector.cap = .01;incumbent = candidate;version += 1
                diagnostics = {key: float(np.mean([m[key] for m in losses])) for key in losses[0]}
                with torch.no_grad():
                    diagnostics.update(anchor_kl_reference=float(anchor_kl(agent.actor, agent.anchor, reference)),
                        std=float(agent.actor.distribution(reference)[1].exp().mean()),
                        added_actor_weight_norm=float(agent.actor.encoder[0].weight[:, 26:].norm()),
                        added_critic_weight_norm=float(agent.critic.q1[0].weight[:, 26:DIMS[args.arm]].norm()))
                append(OUT/f'updates_{run}.csv', dict(env_steps=steps, optimizer_steps=updates,
                    accepted=int(accepted), reasons=';'.join(reasons), rejections=guard.rejections,
                    level=curriculum.level, nominal_candidate_success=candidate['nominal']['policy']['success'],
                    curriculum_candidate_success=candidate[conditions[-1]]['policy']['success'], **diagnostics))
                for key, value in diagnostics.items(): writer.add_scalar(f'diagnostics/{key}', value, steps)
                metrics = formal()
                print(f'{run} step={steps} random={metrics["level2"]["success"]:.4f} nominal={metrics["nominal"]["success"]:.4f} online={successes}/{episodes} level={curriculum.level} reject={guard.rejections}', flush=True)
                next_guard += 10000;losses.clear();guard.begin(agent)
        dataset_folder = ROOT/'datasets'/OUT.name;dataset_folder.mkdir(parents=True, exist_ok=True)
        with h5py.File(dataset_folder/f'{run}.h5', 'x') as h5:
            h5.attrs.update(arm=args.arm, seed=args.seed, observation_dim=DIMS[args.arm], steps=steps,
                source_sha256=hashes['source'], anchor_sha256=hashes['anchor'])
            data = h5.create_group('transitions')
            for key, value in online.data.items(): data.create_dataset(key, data=value[:len(online)].cpu().numpy(), compression='gzip', compression_opts=1)
            h5.create_dataset('episodes_json', data=json.dumps(trace_records), dtype=h5py.string_dtype())
        marker = dict(steps=steps, optimizer_steps=updates, online_successes=successes, online_episodes=episodes,
            rejections=guard.rejections, acceptances=guard.acceptances, best_step=best_step,
            final_level=curriculum.level, curriculum=curriculum.history, cumulative_eval_steps=eval_steps,
            wall_time_s=time.perf_counter()-start_time, physical_reset_checks=env.physical_reset_checks,
            source_hashes=hashes, observation_dim=DIMS[args.arm], anchor_strength=args.strength, anchor_kl_weight=STRENGTHS[args.strength], uniform_expert_replay_fraction=.5,
            expert_sac_draws=updates*128, online_sac_draws=updates*128, extra_bc_draws=updates*256,
            pending_episodes=int((lengths>0).sum()))
        (folder/'completed.json').write_text(json.dumps(marker, indent=2))
    finally:
        if client is not None: client.close()
        writer.close();env.close()

if __name__ == '__main__':
    try: main()
    except BaseException:
        import traceback
        error = traceback.format_exc();(OUT/f'error_{args.arm}_{args.strength}_seed{args.seed}_{os.getpid()}.txt').write_text(error);print(error, flush=True)
        raise
    finally: app.close()
