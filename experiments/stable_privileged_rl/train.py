"""Phase 4 Panda Door GT-window, rollback, shared value and replay ablations.

Task and expert data are unchanged. All Phase 4 outputs have isolated paths.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("E0", "E25", "E50", "E100", "E200",
                                          "E100RB", "E100R1", "E100M", "Q0", "QGT", "QV"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=500000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=10000)
parser.add_argument("--eval-rounds", type=int, default=1)
parser.add_argument("--bc-updates", type=int, default=3000)
parser.add_argument("--offline-critic-updates", type=int, default=1000)
parser.add_argument("--updates-per-vector-step", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--offline-fraction", type=float, default=0.25)
parser.add_argument("--actor-bc-weight", type=float, default=10.0)
parser.add_argument("--demo-file", default="door_dataset/door_expert_1000.h5")
parser.add_argument("--aux-weight", type=float, default=0.1)
parser.add_argument("--rollback-threshold", type=float, default=0.25)
parser.add_argument("--rollback-patience", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
import time
from collections import deque
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks  # noqa: F401

from algorithms.replay import ReplayBuffer, mixed_sample
from auxiliary_learning.gt_prediction import AuxiliarySAC
from auxiliary_learning.multitask_encoder import MultiTaskSAC, SharedEncoderSAC
from checkpoint_manager.rollback import RollbackMonitor
from replay.value_weighted_replay_v2 import ValueWeightedTrajectoryReplay
from experiments.stable_privileged_rl.eval_client import EvalClient
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env, transition_after_step

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "stable_privileged_rl"
LOGS = ROOT / "logs" / "stable_privileged_rl"
CHECKPOINTS = ROOT / "checkpoints" / "stable_privileged_rl"
WINDOWS = {"E0": 0, "E25": 25000, "E50": 50000, "E100": 100000,
           "E200": 200000, "E100RB": 100000, "E100R1": 100000,
           "E100M": 100000,
           "Q0": 0, "QGT": 100000, "QV": 100000}


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
    # Every Phase 4 deployment action is computed from robot observation only.
    return agent.act(robot, deterministic=deterministic)


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


@torch.no_grad()
def diagnose(agent, batch):
    robot, gt, action = batch["robot"], batch["privileged"], batch["action"]
    critic_input = agent.critic_input(robot, gt)
    q1, q2 = agent.critic(critic_input, action)
    _, logp = agent.actor(agent.actor_input(robot, gt))
    q = torch.minimum(q1, q2)
    return float(q.mean()), float(q.var(unbiased=False)), float((q1 - q2).abs().mean()), float(-logp.mean())


def sample_batch(online, demos, mode):
    """Keep the Phase 3 25% expert fraction; weight online data only."""
    if not isinstance(online, ValueWeightedTrajectoryReplay):
        return mixed_sample(online, demos, args.batch_size, args.offline_fraction)
    offline_count = round(args.batch_size * args.offline_fraction)
    online_count = args.batch_size - offline_count
    if not len(online):
        offline_count, online_count = args.batch_size, 0
    parts = []
    if online_count:
        parts.append(online.sample(online_count, mode=mode))
    if offline_count:
        parts.append(demos.sample(offline_count))
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in demos.data}


def main():
    start = time.perf_counter()
    if not 0 <= args.offline_fraction <= 1 or args.steps < 1 or args.num_envs < 1:
        raise ValueError("Invalid online budget, environment count, or offline fraction")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_cfg(args.num_envs, device=args.device or "cuda:0", seed=args.seed)
    env = create_env(cfg)
    run = f"{args.variant}_seed{args.seed}"
    writer = SummaryWriter(str(LOGS / run))
    eval_client = None
    try:
        robot_dim = robot_observation(env).shape[-1]
        gt_dim = env.get_tool_state().shape[-1]
        action_dim = env.action_space.shape[-1]
        if action_dim != 7:
            raise RuntimeError(f"Expected 7 continuous IK actions, got {action_dim}")
        demos = load_demonstrations(ROOT / args.demo_file, env.device)
        if demos.data["robot"].shape[1] != robot_dim or demos.data["privileged"].shape[1] != gt_dim:
            raise RuntimeError("Expert and live state dimensions differ")
        window = WINDOWS[args.variant]
        aux_weight = args.aux_weight if window else 0.0
        if args.variant == "E100M":
            agent = MultiTaskSAC(robot_dim, gt_dim, action_dim, device=env.device,
                                 aux_weight=aux_weight)
        elif args.variant in ("Q0", "QGT", "QV"):
            agent = SharedEncoderSAC(robot_dim, gt_dim, action_dim, device=env.device,
                                     aux_weight=aux_weight)
        else:
            agent = AuxiliarySAC(robot_dim, gt_dim, action_dim, device=env.device,
                                 aux_weight=aux_weight)
        if args.variant in ("Q0", "QGT", "QV"):
            online = ValueWeightedTrajectoryReplay(args.steps + args.num_envs,
                robot_dim, gt_dim, action_dim, args.num_envs, device=env.device,
                gamma=agent.cfg.gamma)
        else:
            online = ReplayBuffer(args.steps + args.num_envs, robot_dim, gt_dim,
                                  action_dim, device=env.device)
        monitor = RollbackMonitor(args.rollback_threshold, patience=args.rollback_patience) \
            if args.variant in ("E100RB", "E100R1") else None

        # All standard A/B actors receive the same initial weights and BC mini-batches.
        torch.manual_seed(args.seed + 100000)
        for update in range(args.bc_updates):
            batch = demos.sample(args.batch_size)
            bc_loss = agent.bc_step(batch["robot"], batch["privileged"], batch["action"])
            if update % 100 == 0:
                writer.add_scalar("bc/loss", bc_loss, update)
        print(f"{run}: BC complete, loss={bc_loss:.5f}, demos={len(demos)}", flush=True)
        if isinstance(agent, SharedEncoderSAC):
            # The policy encoder changed during BC; Q targets must start from
            # the same representation before offline value initialization.
            agent.target_encoder.load_state_dict(agent.actor.encoder.state_dict())
        for update in range(args.offline_critic_updates):
            loss = agent.critic_step(demos.sample(args.batch_size))
            if update % 100 == 0:
                writer.add_scalar("offline/critic_loss", loss, update)
        print(f"{run}: offline critic complete, loss={loss:.5f}", flush=True)

        eval_csv = OUT / f"eval_{run}.csv"
        diag_csv = OUT / f"diagnostics_{run}.csv"
        eval_client = EvalClient(ROOT, args.variant, args.seed, args.num_envs,
                                 args.device or "cuda:0")
        success, mean_return, contact, episodes, eval_steps = eval_client.evaluate(
            agent, 50000 + args.seed, args.eval_rounds)
        if monitor:
            monitor.observe(agent, success)
        append_csv(eval_csv, ("variant", "seed", "env_steps", "success_rate", "mean_return", "contact_rate",
                              "eval_episodes", "eval_env_steps", "wall_time_s", "pre_rollback_success",
                              "rollback", "rollback_count", "best_seen_success", "value_rank_rho", "weighted_replay_active"),
                   (args.variant, args.seed, 0, success, mean_return, contact, episodes, eval_steps,
                    time.perf_counter() - start, success, 0, 0, success, float("nan"), 0))
        save_checkpoint(agent, CHECKPOINTS / run / "step_0.pt", 0)
        writer.add_scalar("eval/success_rate", success, 0)
        writer.add_scalar("eval/mean_return", mean_return, 0)
        env.reset(seed=args.seed)
        episode_returns = torch.zeros(env.num_envs, device=env.device)
        env_steps = 0
        next_eval = args.eval_every
        recent_success = deque(maxlen=256)
        optimizer_steps = 0
        change_logged = window == 0
        while env_steps < args.steps:
            if not change_logged and env_steps >= window:
                agent.aux_weight = 0.0
                for head in ("gt_head", "angle_head", "contact_head"):
                    if hasattr(agent.actor, head):
                        for p in getattr(agent.actor, head).parameters():
                            p.requires_grad_(False)
                change_logged = True
                writer.add_scalar("auxiliary/weight", 0.0, env_steps)
                print(f"{run}: GT auxiliary loss disabled at {env_steps} steps", flush=True)
            robot = robot_observation(env).clone()
            gt = env.get_tool_state().clone()
            action = policy_action(agent, env, deterministic=False)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            transition = {"robot": robot, "privileged": gt, "action": action.detach(),
                          "reward": reward.reshape(-1, 1), "next_robot": next_robot,
                          "next_privileged": next_gt, "done": done.float().reshape(-1, 1)}
            if isinstance(online, ValueWeightedTrajectoryReplay):
                successes = (env.final_door_angle[:, 0] > SUCCESS_ANGLE_RAD).float()
                online.add_step(transition, done, successes, agent)
            else:
                online.add(transition)
            episode_returns += reward
            if done.any():
                finished = done.nonzero(as_tuple=False).squeeze(-1).tolist()
                writer.add_scalar("train/mean_finished_return", float(episode_returns[done].mean()), env_steps)
                writer.add_scalar("train/finished_success_rate", float((env.final_door_angle[done, 0] > SUCCESS_ANGLE_RAD).float().mean()), env_steps)
                recent_success.extend(float(env.final_door_angle[i, 0] > SUCCESS_ANGLE_RAD)
                                      for i in finished)
                episode_returns[done] = 0
            env_steps += env.num_envs
            for _ in range(args.updates_per_vector_step):
                batch = sample_batch(online, demos, "weighted" if args.variant == "QV" else "uniform")
                bc_batch = demos.sample(args.batch_size) if args.actor_bc_weight else None
                metrics = agent.update(batch, bc_batch=bc_batch, bc_weight=args.actor_bc_weight)
                optimizer_steps += 1
            q_mean, q_var, q_disagreement, entropy = diagnose(agent, batch)
            append_csv(diag_csv,
                       ("variant", "seed", "env_steps", "optimizer_steps", "critic_loss",
                        "q_mean", "q_variance", "q_disagreement", "actor_loss", "policy_entropy",
                        "alpha", "online_success_rolling256", "episodes_in_window",
                        "aux_active", "aux_angle_loss", "aux_contact_loss",
                        "value_rank_rho", "weighted_replay_active"),
                       (args.variant, args.seed, env_steps, optimizer_steps,
                        metrics["critic_loss"], q_mean, q_var, q_disagreement,
                        metrics["actor_loss"], entropy, metrics["alpha"],
                        float(np.mean(recent_success)) if recent_success else float("nan"),
                        len(recent_success), int(agent.aux_weight > 0),
                        metrics.get("aux_angle_loss", float("nan")),
                        metrics.get("aux_contact_loss", float("nan")),
                        getattr(online, "calibration_rho", float("nan")),
                        int(args.variant == "QV" and getattr(online, "weighting_active", False))))
            if env_steps % 1024 < env.num_envs:
                for name, value in metrics.items():
                    writer.add_scalar(f"sac/{name}", value, env_steps)
                writer.add_scalar("diagnostics/q_mean", q_mean, env_steps)
                writer.add_scalar("diagnostics/q_variance", q_var, env_steps)
                writer.add_scalar("diagnostics/policy_entropy", entropy, env_steps)
            if env_steps >= next_eval or env_steps >= args.steps:
                success, mean_return, contact, episodes, extra = eval_client.evaluate(
                    agent, 50000 + args.seed, args.eval_rounds)
                eval_steps += extra
                raw_success = success
                rolled_back = False
                if monitor:
                    decision = monitor.observe(agent, success)
                    rolled_back = decision.rollback
                    if rolled_back:
                        success, mean_return, contact, episodes, extra = eval_client.evaluate(
                            agent, 50000 + args.seed, args.eval_rounds)
                        eval_steps += extra
                        print(f"{run}: rollback at {env_steps}, raw={raw_success:.3f}, restored={success:.3f}", flush=True)
                if isinstance(online, ValueWeightedTrajectoryReplay):
                    online.refresh_weights()
                elapsed = time.perf_counter() - start
                append_csv(eval_csv, ("variant", "seed", "env_steps", "success_rate", "mean_return", "contact_rate",
                                      "eval_episodes", "eval_env_steps", "wall_time_s", "pre_rollback_success",
                                      "rollback", "rollback_count", "best_seen_success", "value_rank_rho", "weighted_replay_active"),
                           (args.variant, args.seed, env_steps, success, mean_return, contact, episodes,
                            eval_steps, elapsed, raw_success, int(rolled_back),
                            monitor.rollback_count if monitor else 0,
                            monitor.best_success if monitor else float("nan"),
                            getattr(online, "calibration_rho", float("nan")),
                            int(args.variant == "QV" and getattr(online, "weighting_active", False))))
                writer.add_scalar("eval/success_rate", success, env_steps)
                writer.add_scalar("eval/mean_return", mean_return, env_steps)
                writer.add_scalar("eval/contact_rate", contact, env_steps)
                writer.add_scalar("eval/total_eval_steps", eval_steps, env_steps)
                save_checkpoint(agent, CHECKPOINTS / run / f"step_{env_steps}.pt", env_steps)
                print(f"{run}: steps={env_steps} success={success:.3f} reward={mean_return:.3f}", flush=True)
                next_eval += args.eval_every
        if isinstance(online, ValueWeightedTrajectoryReplay):
            online.export_hdf5(OUT / f"trajectories_{run}.h5")
    finally:
        if eval_client is not None:
            eval_client.close()
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        path = LOGS / f"{args.variant}_seed{args.seed}.error.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(detail, encoding="utf-8")
        print(detail, flush=True)
        raise
    finally:
        app.close()
