"""Train candidates from Phase 8 anchors; commit only after physical evaluation."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=300000)
parser.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks  # noqa
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from algorithms.replay import ReplayBuffer
from offline_rl.baseline import load_demonstrations
from progress_rl.agent import ExpertView
from progress_rl.door_env import config, create
from progress_rl.progress_monitor import EpisodeMonitor
from safe_online.anchor_policy import AnchoredSAC
from safe_online.std_schedule import StdSchedule
from safe_online.safe_update import SafeUpdate
from safe_online.kl_constraint import anchor_kl, feasible_entropy_target
from replay.quality_replay import QualityReplay
from trajectory_quality.quality_model import quality_score
from experiments.phase10_quality_lwd.protocol import ARMS
from experiments.phase10_quality_lwd.eval_client import EvaluationClient
from experiments.phase10_quality_lwd.data import expert_records, save_online
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase10_quality_lwd"
LOG = ROOT/"logs"/"phase10_quality_lwd"
CKPT = ROOT/"checkpoints"/"phase10_quality_lwd"


def write(path, row):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def save(agent, path, steps, metrics, anchor_hash):
    state = agent.checkpoint()
    anchor_path = ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt'
    state.update(env_steps=steps, phase10_arm=args.arm, seed=args.seed,
                 validation=metrics, phase9_source_sha256=anchor_hash,
                 anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest(), arm_config=ARMS[args.arm])
    torch.save(state, path)


def main():
    if args.arm not in ARMS or args.steps % 96 or (not args.smoke and args.steps != 300000):
        raise ValueError("Invalid preregistered arm/budget")
    settings = ARMS[args.arm]
    run = f"P10{args.arm}_seed{args.seed}"+("_smoke" if args.smoke else "")
    folder = CKPT/run
    folder.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source_path = ROOT/"checkpoints"/"phase9_safe_online"/f"P9C_seed{args.seed}"/"step_300000.pt"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = torch.load(source_path, map_location=args.device or "cuda:0", weights_only=False)
    env = create(config("B", 96, args.device or "cuda:0", args.seed))
    client = None
    writer = SummaryWriter(str(LOG/run))
    agent = AnchoredSAC(source, env.device, 1.)
    anchor_path = ROOT/"checkpoints"/"phase8_progress_rl"/f"P8B_seed{args.seed}"/"best.pt"
    anchor_state = torch.load(anchor_path, map_location=env.device, weights_only=False)
    agent.anchor.load_state_dict(anchor_state['actor'])
    schedule = StdSchedule('bound')
    collector = copy.deepcopy(agent.actor).eval()
    collector.requires_grad_(False)
    expert = ExpertView(load_demonstrations(ROOT/"door_dataset"/"door_expert_1000.h5", env.device), "B", env.step_dt)
    online = ReplayBuffer(args.steps+96, 26, 11, 7, env.device)
    replay = QualityReplay(expert, online, expert_records(ROOT/'door_dataset/door_expert_1000.h5'),
                           settings['mode'], settings['temperature'])
    torch.manual_seed(300000+args.seed)
    reference = expert.sample(2048)["robot"].clone()
    guard = SafeUpdate()
    monitor = EpisodeMonitor(env.num_envs, env.device)
    lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    q_sums = torch.zeros_like(returns)
    initial_q = torch.zeros_like(returns)
    discounted_returns = torch.zeros_like(returns)
    contacts = torch.zeros_like(returns)
    moving_ticks = torch.zeros_like(returns)
    start_angles = torch.zeros_like(returns)
    trajectory_starts = torch.arange(env.num_envs, device=env.device)
    episode_first_version = torch.zeros_like(lengths)
    steps = optimizer_steps = online_successes = online_episodes = policy_version = 0
    guard_interval = 10000
    next_guard, next_formal = guard_interval, 10000
    eval_steps = 0
    started = time.perf_counter()
    first_success_step = None
    best_score = None
    best_step = 0
    losses = []
    try:
        client = EvaluationClient(ROOT, args.seed, run)
        cap = schedule.cap(0)
        agent.actor.cap = agent.anchor.cap = collector.cap = cap
        if cap != .01 or source['std_cap'] != .01:
            raise RuntimeError('Phase9 C executed distribution changed')
        incumbent = client.pair(collector, 70000+args.seed)
        eval_steps += sum(m["eval_env_steps"] for m in incumbent.values())

        def formal(candidate, accepted):
            nonlocal eval_steps, best_score, best_step
            metrics = client.pair(agent.actor, 71000+args.seed, rounds=2)
            eval_steps += sum(m["eval_env_steps"] for m in metrics.values())
            for mode in ("deterministic", "policy"):
                write(OUT/f"eval_{run}_{mode}.csv", dict(env_steps=steps, **metrics[mode],
                    raw_candidate_success=candidate[mode]["success"], accepted=int(accepted),
                    rollback_events=guard.rollback_events, accepted_blocks=guard.acceptances,
                    online_successes=online_successes, online_episodes=online_episodes,
                    online_success_ratio=online_successes/max(1, online_episodes),
                    first_online_success_step=first_success_step, cumulative_eval_steps=eval_steps,
                    wall_time_s=time.perf_counter()-started))
            score = guard.score(metrics)
            save(agent, folder/f"step_{steps}.pt", steps, metrics, source_hash)
            if best_score is None or score > best_score:
                best_score, best_step = score, steps
                save(agent, folder/"best.pt", steps, metrics, source_hash)
            for mode in metrics:
                for key, value in metrics[mode].items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f"eval/{mode}/{key}", value, steps)
            writer.add_scalar("online/success_ratio", online_successes/max(1, online_episodes), steps)
            writer.add_scalar("online/successes", online_successes, steps)
            return metrics

        formal(incumbent, True)
        env.reset(seed=args.seed)
        start_angles.copy_(env.get_tool_state()[:, 0])
        torch.manual_seed(400000+args.seed)
        guard.begin(agent)
        while steps < args.steps:
            observation = robot_observation(env).clone()
            gt = env.get_tool_state().clone()
            with torch.no_grad():
                action = collector(observation)[0]
                q1, q2 = agent.critic(observation, action)
                q_sums += torch.minimum(q1,q2).squeeze(-1)
                initial_q = torch.where(lengths==0, torch.minimum(q1,q2).squeeze(-1), initial_q)
            _, reward, terminated, truncated, _ = env.step(action)
            next_observation, next_gt, done = transition_after_step(env, terminated, truncated)
            steps += env.num_envs
            lengths += 1
            returns += reward
            discounted_returns += reward*agent.cfg.gamma**(lengths-1)
            moving = next_gt[:, 0] > start_angles+.02
            moving_ticks += moving
            contacts += moving & (next_gt[:, 9:11]>.5).all(-1)
            monitor.update(gt[:, :1], next_gt[:, :1], (next_gt[:, 9:11]>.5).all(-1, keepdim=True))
            for i in done.nonzero().squeeze(-1).tolist():
                row = monitor.finish(i, next_gt[i, 0], float(env.final_target[i, 0]), lengths[i],
                                     float(env.final_start[i, 0]))
                online_episodes += 1
                online_successes += int(row["success"])
                if row["success"] and first_success_step is None:
                    first_success_step = steps
                row.update(return_value=float(returns[i]), source="fresh_online_exploration",
                    first_policy_version=int(episode_first_version[i]), last_policy_version=policy_version)
                contact_stability = float(contacts[i]/moving_ticks[i].clamp_min(1))
                score = quality_score(row['success'], row['final_angle'], start_angles[i], env.final_target[i,0], contact_stability)
                record = dict(start=int(trajectory_starts[i]), length=int(lengths[i]), stride=96,
                    success=int(row['success']), quality_score=score, contact_stability=contact_stability,
                    start_angle=float(start_angles[i]), final_angle=row['final_angle'],
                    return_value=float(returns[i]), estimated_value=float(q_sums[i]/lengths[i]),
                    initial_q=float(initial_q[i]), discounted_return=float(discounted_returns[i]), env_steps=steps)
                replay.complete(record)
                row.update(quality_score=score, contact_stability=contact_stability,
                           estimated_value=record['estimated_value'], initial_q=record['initial_q'],
                           discounted_return=record['discounted_return'])
                write(OUT/f"episodes_{run}.csv", dict(env_steps=steps, env_index=i, **row))
                trajectory_starts[i] = steps+i
                start_angles[i] = env.get_tool_state()[i, 0]
                q_sums[i] = contacts[i] = moving_ticks[i] = 0
                discounted_returns[i] = 0
                episode_first_version[i] = policy_version
                lengths[i] = 0
                returns[i] = 0
            online.add(dict(robot=observation, privileged=gt, action=action, reward=reward[:, None],
                next_robot=next_observation, next_privileged=next_gt, done=done.float()[:, None]))
            for _ in range(12):
                losses.append(agent.online_update(replay.sample(), expert.sample(256)))
                optimizer_steps += 1
            if steps >= next_guard or steps == args.steps:
                candidate = client.pair(agent.actor, 70000+args.seed)
                eval_steps += sum(m["eval_env_steps"] for m in candidate.values())
                if steps >= next_formal or steps == args.steps:
                    save(agent, folder/f"raw_step_{steps}.pt", steps, candidate, source_hash)
                diagnostics = {key: float(np.mean([m[key] for m in losses])) for key in losses[0]}
                with torch.no_grad():
                    diagnostics.update(effective_anchor_kl=float(anchor_kl(agent.actor, agent.anchor, reference)),
                        raw_anchor_kl=float(anchor_kl(agent.actor, agent.anchor, reference, False)),
                        effective_std=float(agent.actor.distribution(reference)[1].exp().mean()),
                        raw_std=float(agent.actor.distribution(reference, False)[1].exp().mean()),
                        action_drift_mse=float((agent.actor(reference, True)[0]-agent.anchor(reference, True)[0]).square().mean()),
                        target_entropy=float(agent.target_entropy))
                accepted, reasons = guard.commit(agent, incumbent, candidate)
                if accepted:
                    collector.load_state_dict(agent.actor.state_dict())
                    collector.cap = agent.actor.cap
                    incumbent = candidate
                    policy_version += 1
                row = dict(env_steps=steps, optimizer_steps=optimizer_steps,
                    accepted=int(accepted), reasons=";".join(reasons),
                    rollback_events=guard.rollback_events, accepted_blocks=guard.acceptances,
                    candidate_det_success=candidate["deterministic"]["success"],
                    candidate_policy_success=candidate["policy"]["success"],
                    accepted_det_success=incumbent["deterministic"]["success"],
                    accepted_policy_success=incumbent["policy"]["success"],
                    online_successes=online_successes, online_episodes=online_episodes,
                    cap=agent.actor.cap, cumulative_eval_steps=eval_steps, **diagnostics)
                write(OUT/f"updates_{run}.csv", row)
                with (OUT/f'replay_{run}.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(env_steps=steps, **replay.distribution()))+'\n')
                for key, value in diagnostics.items():
                    writer.add_scalar(f"diagnostics/{key}", value, steps)
                if steps >= next_formal or steps == args.steps:
                    metrics = formal(candidate, accepted)
                    print(f"{run} steps={steps} det={metrics['deterministic']['success']:.3f} policy={metrics['policy']['success']:.3f} online={online_successes}/{online_episodes} rejected={guard.rejections} cap={agent.actor.cap}", flush=True)
                    next_formal += 10000
                losses.clear()
                next_guard += guard_interval
                new_cap = schedule.cap(steps)
                if new_cap != agent.actor.cap:
                    agent.actor.cap = agent.anchor.cap = collector.cap = new_cap
                    agent.target_entropy = feasible_entropy_target(agent.anchor, reference)
                    incumbent = client.pair(collector, 70000+args.seed)
                    eval_steps += sum(m["eval_env_steps"] for m in incumbent.values())
                    write(OUT/f"schedule_{run}.csv", dict(env_steps=steps, cap=new_cap,
                        incumbent_policy_success=incumbent["policy"]["success"], target_entropy=agent.target_entropy))
                guard.begin(agent)
        marker = dict(steps=steps, optimizer_steps=optimizer_steps, online_successes=online_successes,
            online_episodes=online_episodes, first_online_success_step=first_success_step,
            rejected_blocks=guard.rejections, accepted_blocks=guard.acceptances,
            rollback_events=guard.rollback_events, best_step=best_step,
            cumulative_eval_steps=eval_steps, replay_distribution=replay.distribution(),
            guard_interval=guard_interval,
            source_anchor_sha256=source_hash, source_anchor_step=source["env_steps"])
        save_online(ROOT/'datasets/phase10_quality_lwd'/f'{run}.h5', online, replay.records,
                    dict(source='fresh_online_exploration', seed=args.seed, arm=args.arm,
                         training_steps=steps, phase9_source_sha256=source_hash, std_cap=.01))
        (folder/"completed.json").write_text(json.dumps(marker, indent=2))
    finally:
        if client is not None:
            client.close()
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        import os
        error = traceback.format_exc()
        (OUT/f'error_{args.arm}_seed{args.seed}_{os.getpid()}.txt').write_text(error)
        print(error, flush=True)
        os._exit(1)
    finally:
        app.close()
