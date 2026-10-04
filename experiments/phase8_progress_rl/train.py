"""Paired progress state/reward interventions with fixed BC+SAC dynamics."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A", "B", "C", "D", "E"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=300000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks  # noqa
from algorithms.replay import ReplayBuffer
from door_env.door import door_angle
from door_env.isaac_env import transition_after_step
from offline_rl.baseline import load_demonstrations
from replay.expert_online_replay import ExpertOnlineReplay
from diagnostics.policy_drift import reference_snapshot, measure
from progress_rl.agent import make_agent, ExpertView
from progress_rl.door_env import config, create, live_observation, terminal_observation
from progress_rl.progress_monitor import EpisodeMonitor, ProgressGuard
from progress_rl.curriculum import ProgressCurriculum
from experiments.phase8_progress_rl.eval_client import ProgressEvalClient
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
LOG = ROOT/"logs"/"phase8_progress_rl"
CKPT = ROOT/"checkpoints"/"phase8_progress_rl"


def write_row(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def save(agent, path, steps, metrics):
    state = agent.checkpoint()
    state.update(env_steps=steps, progress_variant=args.variant, seed=args.seed,
                 training_config=vars(args), validation=metrics)
    torch.save(state, path)


def main():
    if not args.smoke and args.steps not in (300000, 500000):
        raise ValueError("Only preregistered 300k/500k budgets allowed")
    run = f"P8{args.variant}_seed{args.seed}"+("_smoke" if args.smoke else "")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    folder = CKPT/run
    folder.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    env = create(config(args.variant, args.num_envs, args.device or "cuda:0", args.seed))
    curriculum = ProgressCurriculum() if args.variant == "E" else None
    if curriculum:
        env.goal = curriculum.target
        env.reset(seed=args.seed)
    obs_dim = 31 if args.variant in ("C", "D", "E") else 26
    agent = make_agent(args.variant, args.seed, env.device)
    source = load_demonstrations(ROOT/"door_dataset"/"door_expert_1000.h5", env.device)
    expert = ExpertView(source, args.variant, env.step_dt, env.goal)
    torch.manual_seed(args.seed+100000)
    for _ in range(50 if args.smoke else 3000):
        batch = expert.sample(256)
        agent.bc_step(batch["robot"], batch["privileged"], batch["action"])
    agent.reset_optimizer_after_bc()
    agent.set_online_learning_rate()
    online = ReplayBuffer(args.steps+args.num_envs, obs_dim, 11, 7, env.device)
    replay = ExpertOnlineReplay(expert, online, 256, 0.5)
    torch.manual_seed(args.seed+200000)
    reference = reference_snapshot(agent, expert.sample(2048))
    fixed_q = expert.sample(2048)
    writer = SummaryWriter(str(LOG/run))
    client = None
    guard = ProgressGuard()
    monitor = EpisodeMonitor(env.num_envs, env.device)
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    steps = optimizer_steps = online_successes = 0
    next_eval = 10000
    started = time.perf_counter()
    total_eval_steps = 0
    losses = []
    try:
        client = ProgressEvalClient(ROOT, args.variant, args.seed, run, args.num_envs)
        initial = client.evaluate(agent, 50000+args.seed)
        total_eval_steps += initial["eval_env_steps"]
        initial_noise = client.evaluate(agent, 50000+args.seed, noise=0.01)
        total_eval_steps += initial_noise["eval_env_steps"]
        write_row(OUT/f"noise_{run}.csv", dict(env_steps=0, **initial_noise))
        guard.observe(agent, initial, 0)
        save(agent, folder/"best.pt", 0, initial)
        save(agent, folder/"step_0.pt", 0, initial)
        write_row(OUT/f"eval_{run}.csv", dict(env_steps=0, **initial,
                  raw_success=initial["success"], raw_progress=initial["progress"],
                  guard_event="initial", rollbacks=0, online_successes=0,
                  selected_best_success=initial["success"], selected_best_step=0,
                  cumulative_eval_steps=total_eval_steps, wall_time_s=time.perf_counter()-started))
        env.reset(seed=args.seed)
        while steps < args.steps:
            obs = live_observation(env, args.variant).clone()
            gt = env.get_tool_state().clone()
            before = door_angle(env).clone()
            action = agent.act(obs, deterministic=False)
            _, reward, terminated, truncated, _ = env.step(action)
            _, next_gt, done = transition_after_step(env, terminated, truncated)
            nxt = terminal_observation(env, args.variant, done)
            after = next_gt[:, :1]
            contact = (next_gt[:, 9:11] > 0.5).all(dim=-1, keepdim=True)
            monitor.update(before, after, contact)
            episode_lengths += 1
            episode_returns += reward
            steps += env.num_envs
            for i in done.nonzero().squeeze(-1).tolist():
                row = monitor.finish(i, after[i, 0], float(env.final_target[i, 0]), episode_lengths[i],
                                     float(env.final_start[i, 0]))
                row["return"] = float(episode_returns[i])
                online_successes += int(row["success"])
                write_row(OUT/f"episodes_{run}.csv", dict(env_steps=steps, env_index=i,
                          target_angle=float(env.final_target[i, 0]), **row))
                episode_lengths[i] = 0
                episode_returns[i] = 0
            online.add(dict(robot=obs, privileged=gt, action=action.detach(), reward=reward[:, None],
                            next_robot=nxt, next_privileged=next_gt, done=done.float()[:, None]))
            for _ in range(4):
                metrics = agent.online_update(replay.sample(), expert.sample(256), 10.0)
                optimizer_steps += 1
                losses.append(metrics)
            if steps >= next_eval or steps >= args.steps:
                raw = client.evaluate(agent, 50000+args.seed)
                total_eval_steps += raw["eval_env_steps"]
                save(agent, folder/f"raw_step_{steps}.pt", steps, raw)
                raw_diag = measure(agent, reference, fixed_q)
                with torch.no_grad():
                    _, log_std = agent.actor.policy_head(agent.actor.encoder(reference["observation"])).split(7, dim=-1)
                    raw_diag["policy_std_mean"] = float(log_std.clamp(-5.0, 2.0).exp().mean())
                event = guard.observe(agent, raw, steps)
                protected = raw
                if event == "rollback":
                    protected = client.evaluate(agent, 50000+args.seed)
                    total_eval_steps += protected["eval_env_steps"]
                noise = client.evaluate(agent, 50000+args.seed, noise=0.01)
                total_eval_steps += noise["eval_env_steps"]
                write_row(OUT/f"noise_{run}.csv", dict(env_steps=steps, **noise))
                write_row(OUT/f"eval_{run}.csv", dict(env_steps=steps, **protected,
                          raw_success=raw["success"], raw_progress=raw["progress"],
                          guard_event=event, rollbacks=guard.rollbacks, online_successes=online_successes,
                          selected_best_success=guard.best["success"], selected_best_step=guard.step,
                          cumulative_eval_steps=total_eval_steps, wall_time_s=time.perf_counter()-started))
                diagnostic = dict(env_steps=steps, optimizer_steps=optimizer_steps, **raw_diag,
                                  **{key: float(np.mean([m[key] for m in losses])) for key in ("critic_loss", "actor_loss", "bc_loss")})
                write_row(OUT/f"diagnostics_{run}.csv", diagnostic)
                for key, value in diagnostic.items():
                    writer.add_scalar(f"diagnostics/{key}", value, steps)
                for key, value in protected.items():
                    if value is not None:
                        writer.add_scalar(f"eval/{key}", value, steps)
                writer.add_scalar("eval/raw_success", raw["success"], steps)
                writer.add_scalar("eval/noise001_success", noise["success"], steps)
                save(agent, folder/f"step_{steps}.pt", steps, protected)
                if event == "new_best":
                    save(agent, folder/"best.pt", steps, protected)
                if curriculum:
                    stage_metrics = client.evaluate(agent, 50000+args.seed, target=curriculum.target)
                    total_eval_steps += stage_metrics["eval_env_steps"]
                    old_stage, old_target = curriculum.stage, curriculum.target
                    promoted = curriculum.observe(stage_metrics["success"], steps)
                    write_row(OUT/f"curriculum_{run}.csv", dict(env_steps=steps, stage=old_stage,
                              target_angle=old_target, stage_success=stage_metrics["success"], promoted=promoted))
                    if promoted:
                        env.goal = curriculum.target
                        expert.target = curriculum.target
                print(f"{run} steps={steps} raw={raw['success']:.3f} protected={protected['success']:.3f} progress={protected['progress']:.3f} noise={noise['success']:.3f} guard={event} online_successes={online_successes}", flush=True)
                losses.clear()
                next_eval += 10000
        (folder/"completed.json").write_text(json.dumps(dict(steps=steps, online_successes=online_successes,
                    rollbacks=guard.rollbacks, cumulative_eval_steps=total_eval_steps)), encoding="utf-8")
    finally:
        if client is not None:
            client.close()
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
