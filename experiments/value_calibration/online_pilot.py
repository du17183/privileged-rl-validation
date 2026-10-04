"""Paired 100k-step E2 replay pilot, launched only after locked offline gate.

The Door scene, reward, reset, demonstrations, BC, SAC and auxiliary schedule
match Phase 4 E100. Only online transition sampling differs between arms.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("uniform", "quality", "quality_return", "quality_offline"), required=True)
parser.add_argument("--seed", type=int, choices=range(5), required=True)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=10000)
parser.add_argument("--bc-updates", type=int, default=3000)
parser.add_argument("--offline-critic-updates", type=int, default=1000)
parser.add_argument("--updates-per-vector-step", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--offline-fraction", type=float, default=0.25)
parser.add_argument("--actor-bc-weight", type=float, default=10.0)
parser.add_argument("--smoke", action="store_true", help="Isolated 2048-step implementation check")
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

from algorithms.replay import ReplayBuffer
from auxiliary_learning.gt_prediction import AuxiliarySAC
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env, transition_after_step
from experiments.stable_privileged_rl.eval_client import EvalClient
from replay.online_quality_replay import OnlineQualityReplay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"
LOGS = ROOT / "logs" / "value_calibration" / "online_pilot"
CKPT = ROOT / "checkpoints" / "value_calibration" / "online_pilot"


def demonstrations(device):
    with h5py.File(ROOT / "door_dataset" / "door_expert_1000.h5", "r") as h5:
        groups = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
        if len(groups) < 1000:
            raise RuntimeError("The unchanged 1000-episode expert dataset is missing")
        count = sum(len(group["action"]) for group in groups)
        first = groups[0]
        buffer = ReplayBuffer(count, first["observation"].shape[1],
                              first["state"].shape[1], first["action"].shape[1], device)
        for group in groups:
            if not bool(group.attrs["success"]):
                raise RuntimeError(f"Expert failure in {group.name}")
            buffer.add({"robot": group["observation"][:], "privileged": group["state"][:],
                        "action": group["action"][:], "reward": group["reward"][:],
                        "next_robot": group["next_observation"][:],
                        "next_privileged": group["next_state"][:], "done": group["done"][:]})
    return buffer


def append_row(path, header, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(header)
        writer.writerow(values)


def save(agent, run, step):
    path = CKPT / run / f"step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = agent.checkpoint()
    state.update({"env_steps": step, "seed": args.seed, "arm": args.arm,
                  "training_config": vars(args).copy(), "pilot_only": True})
    torch.save(state, path)


def batch_sample(online, demos):
    offline_count = round(args.batch_size * args.offline_fraction)
    online_count = args.batch_size - offline_count
    if not len(online):
        online_count, offline_count = 0, args.batch_size
    chunks = []
    if online_count:
        chunks.append(online.sample(online_count,
                                    mode="weighted" if args.arm != "uniform" else "uniform"))
    if offline_count:
        chunks.append(demos.sample(offline_count))
    return {key: torch.cat([part[key] for part in chunks], dim=0) for key in demos.data}


def main():
    if args.smoke:
        if args.steps != 2048 or args.eval_every != 2048 or args.num_envs != 32:
            raise ValueError("Smoke check uses exactly 2048 steps and 32 environments")
    elif args.steps != 100000 or args.num_envs != 32 or args.eval_every != 10000:
        raise ValueError("The registered pilot uses exactly 100k steps, 32 environments, 10k evaluation")
    if not (OUT / "offline_gate_passed.json").exists():
        raise RuntimeError("Offline model gate must be verified before any RL pilot starts")
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = create_env(make_cfg(args.num_envs, device=args.device or "cuda:0", seed=args.seed))
    run = f"{'smoke_' if args.smoke else ''}{args.arm}_seed{args.seed}"
    writer = SummaryWriter(str(LOGS / run))
    evaluator = None
    try:
        robot_dim = robot_observation(env).shape[-1]
        gt_dim = env.get_tool_state().shape[-1]
        action_dim = env.action_space.shape[-1]
        if (robot_dim, gt_dim, action_dim) != (26, 11, 7):
            raise RuntimeError("Door state/action interface changed")
        demos = demonstrations(env.device)
        agent = AuxiliarySAC(robot_dim, gt_dim, action_dim,
                             device=env.device, aux_weight=0.1)
        model_path = (ROOT / "checkpoints" / "value_calibration" /
                      f"EARLY100_seed{args.seed}.pt") if args.arm != "uniform" else None
        online = OnlineQualityReplay(args.steps + args.num_envs, robot_dim, gt_dim,
                                     action_dim, args.num_envs, model_path, device=env.device,
                                     gate_mode={"uniform": "success", "quality": "success",
                                                "quality_return": "return",
                                                "quality_offline": "offline"}[args.arm])
        torch.manual_seed(args.seed + 100000)
        for _ in range(args.bc_updates):
            batch = demos.sample(args.batch_size)
            agent.bc_step(batch["robot"], batch["privileged"], batch["action"])
        for _ in range(args.offline_critic_updates):
            agent.critic_step(demos.sample(args.batch_size))
        evaluator_label = ({"uniform": "P5SU", "quality": "P5SW",
                            "quality_return": "P5SR", "quality_offline": "P5SO"}[args.arm] if args.smoke else
                           {"uniform": "P5U", "quality": "P5W",
                            "quality_return": "P5R", "quality_offline": "P5O"}[args.arm])
        evaluator = EvalClient(ROOT, evaluator_label,
                               args.seed, args.num_envs, args.device or "cuda:0")
        eval_path = OUT / f"eval_{run}.csv"
        header = ("arm", "seed", "env_steps", "success_rate", "mean_return", "contact_rate",
                  "eval_episodes", "eval_env_steps", "complete_online_episodes",
                  "online_rank_auc", "online_return_rho", "weighted_active", "wall_time_s")
        total_eval_steps = 0
        success, reward, contact, episodes, extra = evaluator.evaluate(agent, 50000 + args.seed, 1)
        total_eval_steps += extra
        append_row(eval_path, header, (args.arm, args.seed, 0, success, reward, contact,
                                       episodes, total_eval_steps, online.completed,
                                       online.online_rank_auc, online.online_return_rho,
                                       0, time.perf_counter() - start))
        writer.add_scalar("eval/success_rate", success, 0)
        save(agent, run, 0)
        env.reset(seed=args.seed)
        env_steps = 0
        next_eval = args.eval_every
        while env_steps < args.steps:
            robot = robot_observation(env).clone()
            gt = env.get_tool_state().clone()
            action = agent.act(robot, deterministic=False)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            transition = {"robot": robot, "privileged": gt, "action": action.detach(),
                          "reward": reward.reshape(-1, 1), "next_robot": next_robot,
                          "next_privileged": next_gt, "done": done.float().reshape(-1, 1)}
            successes = (env.final_door_angle[:, 0] > SUCCESS_ANGLE_RAD)
            online.add_step(transition, done, successes, env_steps + env.num_envs)
            env_steps += env.num_envs
            for _ in range(args.updates_per_vector_step):
                batch = batch_sample(online, demos)
                bc_batch = demos.sample(args.batch_size)
                metrics = agent.update(batch, bc_batch=bc_batch,
                                       bc_weight=args.actor_bc_weight)
            if env_steps % 1024 < env.num_envs:
                for key, value in metrics.items():
                    writer.add_scalar(f"sac/{key}", value, env_steps)
                writer.add_scalar("replay/online_rank_auc", online.online_rank_auc, env_steps)
                writer.add_scalar("replay/online_return_rho", online.online_return_rho, env_steps)
                writer.add_scalar("replay/weighted_active", int(online.active), env_steps)
            if env_steps >= next_eval or env_steps >= args.steps:
                success, mean_return, contact, episodes, extra = evaluator.evaluate(
                    agent, 50000 + args.seed, 1)
                total_eval_steps += extra
                online.refresh_weights()
                append_row(eval_path, header, (args.arm, args.seed, env_steps,
                                               success, mean_return, contact, episodes,
                                               total_eval_steps, online.completed,
                                               online.online_rank_auc, online.online_return_rho,
                                               int(online.active),
                                               time.perf_counter() - start))
                writer.add_scalar("eval/success_rate", success, env_steps)
                writer.add_scalar("eval/mean_return", mean_return, env_steps)
                writer.add_scalar("replay/complete_episodes", online.completed, env_steps)
                save(agent, run, env_steps)
                print(f"{run}: steps={env_steps} success={success:.3f} "
                      f"online_auc={online.online_rank_auc:.3f} "
                      f"return_rho={online.online_return_rho:.3f} "
                      f"weighted={online.active}", flush=True)
                next_eval += args.eval_every
        online.export_hdf5(OUT / f"trajectories_{run}.h5")
    finally:
        if evaluator is not None:
            evaluator.close()
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{args.arm}_seed{args.seed}.error.txt").write_text(detail, encoding="utf-8")
        print(detail, flush=True)
        raise
    finally:
        app.close()
