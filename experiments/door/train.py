"""Panda Door BC initialization followed by online A/B/C/D SAC.

A: robot actor, robot critic, uniform online replay.
B: robot actor, robot+GT critic, uniform online replay.
C: robot+GT actor and critic, uniform online replay (oracle bound).
D: B networks, critic-value trajectory weighted online replay.

All variants share the same demonstrations, physical task, reward, update
budget, random seed schedule and held-out evaluation episodes.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A", "B", "C", "D"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=500000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=25000)
parser.add_argument("--eval-rounds", type=int, default=2)
parser.add_argument("--bc-updates", type=int, default=3000)
parser.add_argument("--offline-critic-updates", type=int, default=1000)
parser.add_argument("--updates-per-vector-step", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--offline-fraction", type=float, default=0.25)
parser.add_argument("--actor-bc-weight", type=float, default=10.0)
parser.add_argument("--demo-file", default="door_dataset/door_expert_1000.h5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks  # noqa: F401

from algorithms.asymmetric_sac import AsymmetricSAC
from algorithms.replay import ReplayBuffer, mixed_sample
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env, transition_after_step
from replay.value_weighted_replay import ValueWeightedReplay

ROOT = Path(__file__).resolve().parents[2]


def load_demonstrations(path, device):
    with h5py.File(path, "r") as h5:
        groups = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
        if len(groups) < 1000:
            raise ValueError(f"Expected 1000 expert trajectories; found {len(groups)}")
        required = ("observation", "state", "action", "reward", "next_observation", "next_state", "done")
        total = sum(len(group["action"]) for group in groups)
        buffer = ReplayBuffer(total, groups[0]["observation"].shape[1], groups[0]["state"].shape[1],
                              groups[0]["action"].shape[1], device=device)
        for group in groups:
            if not bool(group.attrs["success"]) or not all(k in group for k in required):
                raise ValueError(f"Invalid demonstration: {group.name}")
            buffer.add({
                "robot": group["observation"][:], "privileged": group["state"][:],
                "action": group["action"][:], "reward": group["reward"][:],
                "next_robot": group["next_observation"][:],
                "next_privileged": group["next_state"][:], "done": group["done"][:],
            })
    return buffer


@torch.no_grad()
def policy_action(agent, env, deterministic):
    robot = robot_observation(env)
    if agent.variant == "C":
        return agent.act(robot, env.get_tool_state(), deterministic=deterministic)
    # For A/B/D there is no call path from fixture GT to the policy.
    return agent.act(robot, deterministic=deterministic)


@torch.no_grad()
def evaluate(agent, env, seed, rounds):
    env.reset(seed=seed)
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    ever_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success, total_returns, contact_rates = [], [], []
    steps = 0
    max_vec_steps = int(env.cfg.episode_length_s / env.step_dt + 2) * (rounds + 1)
    for _ in range(max_vec_steps):
        action = policy_action(agent, env, deterministic=True)
        _, reward, terminated, truncated, _ = env.step(action)
        steps += env.num_envs
        returns += reward
        done = terminated | truncated
        gt = env.get_tool_state()
        contact = (gt[:, -2:] > 0.5).all(dim=-1)
        if done.any():
            contact[done] = (env.final_privileged[done, -2:] > 0.5).all(dim=-1)
        ever_contact |= contact
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            if counts[i] < rounds:
                success.append(float(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD))
                total_returns.append(float(returns[i]))
                contact_rates.append(float(ever_contact[i]))
                counts[i] += 1
            returns[i] = 0
            ever_contact[i] = False
        if bool(torch.all(counts >= rounds)):
            break
    expected = env.num_envs * rounds
    if len(success) != expected:
        raise RuntimeError(f"Evaluation collected {len(success)}/{expected} episodes")
    return float(np.mean(success)), float(np.mean(total_returns)), float(np.mean(contact_rates)), expected, steps


def append_csv(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(header)
        writer.writerow(row)


def save_checkpoint(agent, path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = agent.checkpoint()
    state.update({"env_steps": steps, "seed": args.seed, "variant_label": args.variant,
                  "success_angle_rad": SUCCESS_ANGLE_RAD, "training_config": vars(args).copy()})
    torch.save(state, path)


def main():
    start = time.perf_counter()
    if not 0 <= args.offline_fraction <= 1 or args.steps < 1 or args.num_envs < 1:
        raise ValueError("Invalid online budget, environment count, or offline fraction")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=args.seed)
    env = create_env(cfg)
    run = f"{args.variant}_seed{args.seed}"
    writer = SummaryWriter(str(ROOT / "logs" / "door" / run))
    try:
        robot_dim = robot_observation(env).shape[-1]
        gt_dim = env.get_tool_state().shape[-1]
        action_dim = env.action_space.shape[-1]
        if action_dim != 7:
            raise RuntimeError(f"Expected 7 continuous IK actions, got {action_dim}")
        demos = load_demonstrations(ROOT / args.demo_file, env.device)
        if demos.data["robot"].shape[1] != robot_dim or demos.data["privileged"].shape[1] != gt_dim:
            raise RuntimeError("Expert and live state dimensions differ")
        network_variant = "B" if args.variant == "D" else args.variant
        agent = AsymmetricSAC(robot_dim, gt_dim, action_dim, network_variant, device=env.device)
        if args.variant == "D":
            online = ValueWeightedReplay(args.steps + args.num_envs, robot_dim, gt_dim, action_dim,
                                         device=env.device)
        else:
            online = ReplayBuffer(args.steps + args.num_envs, robot_dim, gt_dim, action_dim, device=env.device)

        # A/B/D actors receive the same initial weights and BC mini-batch sequence.
        torch.manual_seed(args.seed + 100000)
        for update in range(args.bc_updates):
            batch = demos.sample(args.batch_size)
            bc_loss = agent.bc_step(batch["robot"], batch["privileged"], batch["action"])
            if update % 100 == 0:
                writer.add_scalar("bc/loss", bc_loss, update)
        print(f"{run}: BC complete, loss={bc_loss:.5f}, demos={len(demos)}", flush=True)
        for update in range(args.offline_critic_updates):
            loss = agent.critic_step(demos.sample(args.batch_size))
            if update % 100 == 0:
                writer.add_scalar("offline/critic_loss", loss, update)
        print(f"{run}: offline critic complete, loss={loss:.5f}", flush=True)

        eval_csv = ROOT / "results" / "door" / f"eval_{run}.csv"
        replay_csv = ROOT / "results" / "door" / f"replay_{run}.csv"
        success, mean_return, contact, episodes, eval_steps = evaluate(agent, env, 50000 + args.seed, args.eval_rounds)
        append_csv(eval_csv, ("variant", "seed", "env_steps", "success_rate", "mean_return", "contact_rate",
                              "eval_episodes", "eval_env_steps", "wall_time_s"),
                   (args.variant, args.seed, 0, success, mean_return, contact, episodes, eval_steps,
                    time.perf_counter() - start))
        save_checkpoint(agent, ROOT / "checkpoints" / "door" / run / "step_0.pt", 0)
        writer.add_scalar("eval/success_rate", success, 0)
        writer.add_scalar("eval/mean_return", mean_return, 0)
        env.reset(seed=args.seed)
        episode_returns = torch.zeros(env.num_envs, device=env.device)
        env_steps = 0
        next_eval = args.eval_every
        current_ids = list(range(env.num_envs))
        next_id = env.num_envs
        active_indices = [[] for _ in range(env.num_envs)]
        while env_steps < args.steps:
            robot = robot_observation(env).clone()
            gt = env.get_tool_state().clone()
            action = policy_action(agent, env, deterministic=False)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            transition = {"robot": robot, "privileged": gt, "action": action.detach(),
                          "reward": reward.reshape(-1, 1), "next_robot": next_robot,
                          "next_privileged": next_gt, "done": done.float().reshape(-1, 1)}
            if args.variant == "D":
                indices = online.add(transition, current_ids)
                for i, index in enumerate(indices):
                    active_indices[i].append(index)
            else:
                online.add(transition)
            episode_returns += reward
            if done.any():
                finished = done.nonzero(as_tuple=False).squeeze(-1).tolist()
                writer.add_scalar("train/mean_finished_return", float(episode_returns[done].mean()), env_steps)
                writer.add_scalar("train/finished_success_rate", float((env.final_door_angle[done, 0] > SUCCESS_ANGLE_RAD).float().mean()), env_steps)
                for i in finished:
                    if args.variant == "D":
                        online.finalize(current_ids[i], active_indices[i], agent.critic,
                                        bool(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD))
                        active_indices[i] = []
                        current_ids[i] = next_id
                        next_id += 1
                episode_returns[done] = 0
            env_steps += env.num_envs
            for _ in range(args.updates_per_vector_step):
                batch = mixed_sample(online, demos, args.batch_size, args.offline_fraction)
                bc_batch = demos.sample(args.batch_size) if args.actor_bc_weight else None
                metrics = agent.update(batch, bc_batch=bc_batch, bc_weight=args.actor_bc_weight)
            if env_steps % 1024 < env.num_envs:
                for name, value in metrics.items():
                    writer.add_scalar(f"sac/{name}", value, env_steps)
            if env_steps >= next_eval or env_steps >= args.steps:
                success, mean_return, contact, episodes, extra = evaluate(agent, env, 50000 + args.seed, args.eval_rounds)
                eval_steps += extra
                elapsed = time.perf_counter() - start
                append_csv(eval_csv, ("variant", "seed", "env_steps", "success_rate", "mean_return", "contact_rate",
                                      "eval_episodes", "eval_env_steps", "wall_time_s"),
                           (args.variant, args.seed, env_steps, success, mean_return, contact, episodes,
                            eval_steps, elapsed))
                writer.add_scalar("eval/success_rate", success, env_steps)
                writer.add_scalar("eval/mean_return", mean_return, env_steps)
                writer.add_scalar("eval/contact_rate", contact, env_steps)
                writer.add_scalar("eval/total_eval_steps", eval_steps, env_steps)
                if args.variant == "D":
                    stats = online.sampling_stats()
                    append_csv(replay_csv, ("env_steps",) + tuple(stats),
                               (env_steps,) + tuple(stats.values()))
                    for name, value in stats.items():
                        writer.add_scalar(f"replay/{name}", value, env_steps)
                save_checkpoint(agent, ROOT / "checkpoints" / "door" / run / f"step_{env_steps}.pt", env_steps)
                print(f"{run}: steps={env_steps} success={success:.3f} reward={mean_return:.3f}", flush=True)
                next_eval += args.eval_every
                env.reset(seed=args.seed + env_steps)
                episode_returns.zero_()
                if args.variant == "D":
                    active_indices = [[] for _ in range(env.num_envs)]
                    current_ids = list(range(next_id, next_id + env.num_envs))
                    next_id += env.num_envs
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        path = ROOT / "logs" / "door" / f"{args.variant}_seed{args.seed}.error.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(detail, encoding="utf-8")
        print(detail, flush=True)
        raise
    finally:
        app.close()
