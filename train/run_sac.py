"""BC initialization followed by online SAC for A, B, or C.

Example (after Isaac Lab installation):
PYTHONPATH=. .venv/bin/python train/run_sac.py --variant B --seed 0 --steps 200000 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=200000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=10000)
parser.add_argument("--eval-rounds", type=int, default=2)
parser.add_argument("--bc-updates", type=int, default=3000)
parser.add_argument("--updates-per-vector-step", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--offline-fraction", type=float, default=0.25)
parser.add_argument("--offline-critic-updates", type=int, default=0)
parser.add_argument("--actor-bc-weight", type=float, default=0.0)
parser.add_argument("--bc-anneal-steps", type=int, default=0)
parser.add_argument("--demo-file", default="datasets/drawer_expert.h5")
parser.add_argument("--bc-augmentation-file", default=None)
parser.add_argument("--bc-augmentation-fraction", type=float, default=0.5)
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

import isaaclab_tasks  # noqa: F401; registers task

from algorithms.asymmetric_sac import AsymmetricSAC
from algorithms.replay import ReplayBuffer, mixed_sample
from envs.drawer import SUCCESS_DISTANCE_M, make_cfg, robot_observation
from envs.isaac_env import create_env, transition_after_step


ROOT = Path(__file__).resolve().parents[1]


def load_demonstrations(path, device):
    with h5py.File(path, "r") as h5:
        groups = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
        if len(groups) < 500:
            raise ValueError(f"Expected >=500 successful expert trajectories, found {len(groups)}")
        required = ("observation", "state", "action", "reward", "next_observation", "next_state", "done")
        total = sum(len(group["action"]) for group in groups)
        rdim = groups[0]["observation"].shape[1]
        pdim = groups[0]["state"].shape[1]
        adim = groups[0]["action"].shape[1]
        buffer = ReplayBuffer(total, rdim, pdim, adim, device=device)
        for group in groups:
            if not bool(group.attrs["success"]):
                raise ValueError("Dataset contains a failed demonstration")
            if not all(key in group for key in required):
                raise ValueError(f"Missing dataset field in {group.name}")
            batch = {
                "robot": group["observation"][:], "privileged": group["state"][:],
                "action": group["action"][:], "reward": group["reward"][:],
                "next_robot": group["next_observation"][:],
                "next_privileged": group["next_state"][:], "done": group["done"][:],
            }
            buffer.add(batch)
    return buffer


def load_bc_augmentation(path, device, robot_dim, privileged_dim, action_dim):
    with h5py.File(path, "r") as h5:
        fields = ("observation", "state", "action")
        if not all(key in h5 for key in fields):
            raise ValueError(f"BC augmentation missing fields: {path}")
        values = [torch.as_tensor(h5[key][:], device=device, dtype=torch.float32) for key in fields]
    if [item.shape[1] for item in values] != [robot_dim, privileged_dim, action_dim]:
        raise ValueError("BC augmentation shape differs from environment")
    if len({len(item) for item in values}) != 1:
        raise ValueError("BC augmentation columns have different lengths")
    return dict(zip(("robot", "privileged", "action"), values))


def sample_bc_batch(demos, augmentation, batch_size, fraction):
    count_aug = int(round(batch_size * fraction)) if augmentation is not None else 0
    count_demo = batch_size - count_aug
    batch = demos.sample(count_demo) if count_demo else {}
    if count_aug:
        idx = torch.randint(len(augmentation["robot"]), (count_aug,), device=augmentation["robot"].device)
        for key, value in augmentation.items():
            batch[key] = torch.cat((batch[key], value[idx]), dim=0) if count_demo else value[idx]
    return batch


@torch.no_grad()
def policy_action(agent, env, deterministic):
    robot = robot_observation(env)
    if agent.variant == "C":
        return agent.act(robot, env.get_tool_state(), deterministic=deterministic)
    # A/B evaluation and deployment have no path to environment GT.
    return agent.act(robot, deterministic=deterministic)


@torch.no_grad()
def evaluate(agent, env, seed, rounds):
    env.reset(seed=seed)
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    ever_grasp = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    successes, episode_returns, grasps, grasped_successes = [], [], [], []
    max_steps = int(env.cfg.episode_length_s / env.step_dt + 2) * (rounds + 1)
    eval_steps = 0
    for _ in range(max_steps):
        action = policy_action(agent, env, deterministic=True)
        _, reward, terminated, truncated, _ = env.step(action)
        eval_steps += env.num_envs
        returns += reward
        done = terminated | truncated
        gt = env.get_tool_state()
        contact = (gt[:, -2:] > 0.5).all(dim=-1)
        if done.any():
            contact[done] = (env.final_privileged[done, -2:] > 0.5).all(dim=-1)
        ever_grasp |= contact
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            if counts[i] < rounds:
                succeeded = bool(env.final_drawer_position[i, 0] > SUCCESS_DISTANCE_M)
                successes.append(float(succeeded))
                episode_returns.append(float(returns[i]))
                grasps.append(float(ever_grasp[i]))
                grasped_successes.append(float(succeeded and ever_grasp[i]))
                counts[i] += 1
            returns[i] = 0
            ever_grasp[i] = False
        if bool(torch.all(counts >= rounds)):
            break
    expected = rounds * env.num_envs
    if len(successes) != expected:
        raise RuntimeError(f"Evaluation collected {len(successes)}/{expected} episodes")
    return (float(np.mean(successes)), float(np.mean(episode_returns)),
            float(np.mean(grasps)), float(np.mean(grasped_successes)), expected, eval_steps)


def append_eval(csv_path, variant, seed, steps, success_rate, mean_return, grasp_rate,
                grasped_success_rate, episodes, eval_env_steps, wall_time_s):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(("variant", "seed", "env_steps", "success_rate", "mean_return",
                             "grasp_rate", "contact_assisted_success_rate", "eval_episodes",
                             "eval_env_steps", "wall_time_s"))
        writer.writerow((variant, seed, steps, success_rate, mean_return, grasp_rate,
                         grasped_success_rate, episodes, eval_env_steps, wall_time_s))


def save_checkpoint(agent, path, steps, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = agent.checkpoint()
    state.update({
        "env_steps": steps, "seed": seed, "success_distance_m": SUCCESS_DISTANCE_M,
        "training_config": vars(args).copy(),
    })
    torch.save(state, path)


def main():
    start_time = time.perf_counter()
    if not 0 <= args.offline_fraction <= 1:
        raise ValueError("offline-fraction must be in [0,1]")
    if args.offline_critic_updates < 0 or args.actor_bc_weight < 0 or args.bc_anneal_steps < 0:
        raise ValueError("Offline updates, BC weight, and anneal steps must be nonnegative")
    if not 0 <= args.bc_augmentation_fraction <= 1:
        raise ValueError("bc-augmentation-fraction must be in [0,1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=args.seed)
    env = create_env(cfg)
    run_name = f"{args.variant}_seed{args.seed}"
    writer = SummaryWriter(str(ROOT / "logs" / run_name))
    try:
        robot_dim = robot_observation(env).shape[-1]
        privileged_dim = env.get_tool_state().shape[-1]
        action_dim = env.action_space.shape[-1]
        if action_dim != 7:
            raise RuntimeError(f"Expected relative IK (6) + gripper (1), got {action_dim}")
        demos = load_demonstrations(ROOT / args.demo_file, env.device)
        if demos.data["robot"].shape[-1] != robot_dim or demos.data["privileged"].shape[-1] != privileged_dim:
            raise ValueError("Demo observation/state shape differs from live environment")
        agent = AsymmetricSAC(robot_dim, privileged_dim, action_dim, args.variant, device=env.device)
        augmentation = None
        if args.bc_augmentation_file:
            augmentation = load_bc_augmentation(
                ROOT / args.bc_augmentation_file, env.device, robot_dim, privileged_dim, action_dim
            )
            print(f"{run_name}: loaded {len(augmentation['robot'])} corrective BC labels", flush=True)
        online = ReplayBuffer(500000, robot_dim, privileged_dim, action_dim, device=env.device)

        # A and B initialize identical actors and consume identical BC batches per seed.
        torch.manual_seed(args.seed + 100000)
        for update in range(args.bc_updates):
            batch = sample_bc_batch(demos, augmentation, args.batch_size, args.bc_augmentation_fraction)
            loss = agent.bc_step(batch["robot"], batch["privileged"], batch["action"])
            if update % 100 == 0:
                writer.add_scalar("bc/loss", loss, update)
        print(f"{run_name}: BC complete, loss={loss:.5f}", flush=True)

        for update in range(args.offline_critic_updates):
            critic_loss = agent.critic_step(demos.sample(args.batch_size))
            if update % 100 == 0:
                writer.add_scalar("offline/critic_loss", critic_loss, update)
        if args.offline_critic_updates:
            print(f"{run_name}: offline critic complete, loss={critic_loss:.5f}", flush=True)

        eval_csv = ROOT / "results" / f"eval_{run_name}.csv"
        success_rate, mean_return, grasp_rate, grasped_success_rate, episodes, eval_steps = evaluate(
            agent, env, 50000 + args.seed, args.eval_rounds
        )
        append_eval(eval_csv, args.variant, args.seed, 0, success_rate, mean_return,
                    grasp_rate, grasped_success_rate, episodes, eval_steps, time.perf_counter() - start_time)
        save_checkpoint(agent, ROOT / "checkpoints" / run_name / "step_0.pt", 0, args.seed)
        writer.add_scalar("eval/success_rate", success_rate, 0)
        writer.add_scalar("eval/mean_return", mean_return, 0)
        writer.add_scalar("eval/grasp_rate", grasp_rate, 0)
        writer.add_scalar("eval/contact_assisted_success_rate", grasped_success_rate, 0)
        env.reset(seed=args.seed)
        episode_return = torch.zeros(env.num_envs, device=env.device)
        env_steps = 0
        next_eval = args.eval_every
        while env_steps < args.steps:
            robot = robot_observation(env).clone()
            gt = env.get_tool_state().clone()
            action = policy_action(agent, env, deterministic=False)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            online.add({
                "robot": robot, "privileged": gt, "action": action.detach(),
                "reward": reward.reshape(-1, 1), "next_robot": next_robot,
                "next_privileged": next_gt, "done": done.float().reshape(-1, 1),
            })
            episode_return += reward
            if done.any():
                writer.add_scalar("train/mean_finished_return", float(episode_return[done].mean()), env_steps)
                writer.add_scalar("train/finished_success_rate", float((env.final_drawer_position[done, 0] > SUCCESS_DISTANCE_M).float().mean()), env_steps)
                episode_return[done] = 0
            env_steps += env.num_envs
            bc_weight = args.actor_bc_weight
            if args.bc_anneal_steps:
                bc_weight *= max(0.0, 1.0 - env_steps / args.bc_anneal_steps)
            for _ in range(args.updates_per_vector_step):
                batch = mixed_sample(online, demos, args.batch_size, args.offline_fraction)
                bc_batch = sample_bc_batch(demos, augmentation, args.batch_size, args.bc_augmentation_fraction) if bc_weight else None
                metrics = agent.update(batch, bc_batch=bc_batch, bc_weight=bc_weight)
            if env_steps % 1024 < env.num_envs:
                for name, value in metrics.items():
                    writer.add_scalar(f"sac/{name}", value, env_steps)
            if env_steps >= next_eval or env_steps >= args.steps:
                success_rate, mean_return, grasp_rate, grasped_success_rate, episodes, extra = evaluate(
                    agent, env, 50000 + args.seed, args.eval_rounds
                )
                eval_steps += extra
                elapsed_s = time.perf_counter() - start_time
                append_eval(eval_csv, args.variant, args.seed, env_steps, success_rate, mean_return,
                            grasp_rate, grasped_success_rate, episodes, eval_steps, elapsed_s)
                writer.add_scalar("eval/success_rate", success_rate, env_steps)
                writer.add_scalar("eval/mean_return", mean_return, env_steps)
                writer.add_scalar("eval/grasp_rate", grasp_rate, env_steps)
                writer.add_scalar("eval/contact_assisted_success_rate", grasped_success_rate, env_steps)
                writer.add_scalar("eval/total_eval_steps", eval_steps, env_steps)
                writer.add_scalar("time/wall_hours", elapsed_s / 3600.0, env_steps)
                save_checkpoint(agent, ROOT / "checkpoints" / run_name / f"step_{env_steps}.pt", env_steps, args.seed)
                print(f"{run_name}: steps={env_steps} success={success_rate:.3f} return={mean_return:.3f}", flush=True)
                next_eval += args.eval_every
                env.reset(seed=args.seed + env_steps)
                episode_return.zero_()
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        detail = traceback.format_exc()
        error_path = ROOT / "logs" / f"{args.variant}_seed{args.seed}.error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(detail, encoding="utf-8")
        print(detail, flush=True)
        raise
    finally:
        app.close()
