"""Phase 7 paired low-temperature Door trials with an explicit alpha schedule."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("E1L", "E2L"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=300000)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=10000)
parser.add_argument("--eval-rounds", type=int, default=2)
parser.add_argument("--bc-updates", type=int, default=3000)
parser.add_argument("--updates-per-vector-step", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--demo-file", default="door_dataset/door_expert_1000.h5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import csv
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks  # noqa: F401

from algorithms.replay import ReplayBuffer
from diagnostics.policy_drift import measure, reference_snapshot
from door_env.door import SUCCESS_ANGLE_RAD, make_cfg, robot_observation
from door_env.isaac_env import create_env, transition_after_step
from experiments.stable_privileged_rl.eval_client import EvalClient
from offline_rl.baseline import load_demonstrations
from regularization.bc_regularized_sac import Phase6SAC
from replay.expert_online_replay import ExpertOnlineReplay
from stability.entropy_schedule import fixed_alpha, sample_rollout_action
from stability.bc_regularization import BC_WEIGHTS
from stability.checkpoint_guard import CheckpointGuard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
LOGS = ROOT / "logs" / "phase7_stable_policy"
CHECKPOINTS = ROOT / "checkpoints" / "phase7_stable_policy"


def append_csv(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(header)
        writer.writerow(row)


def save_checkpoint(agent, path, step, recipe):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = agent.checkpoint()
    state.update({"env_steps": step, "seed": args.seed,
                  "variant_label": f"P7{args.variant}", "recipe": recipe,
                  "training_config": vars(args).copy(),
                  "success_angle_rad": SUCCESS_ANGLE_RAD})
    torch.save(state, path)


def main():
    if args.steps < 300000 or args.steps > 500000 or args.num_envs < 1:
        raise ValueError("Phase 7 training budget must be in 300000..500000")
    if args.eval_rounds * args.num_envs < 50:
        raise ValueError("At least 50 episodes are required per evaluation")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or "cuda:0"
    env = create_env(make_cfg(args.num_envs, device=device, seed=args.seed))
    run = f"P7{args.variant}_seed{args.seed}"
    log_path = LOGS / run
    writer = SummaryWriter(str(log_path))
    eval_client = None
    try:
        robot_dim = robot_observation(env).shape[-1]
        gt_dim = env.get_tool_state().shape[-1]
        action_dim = env.action_space.shape[-1]
        demos = load_demonstrations(ROOT / args.demo_file, env.device)
        if (robot_dim, gt_dim, action_dim) != (26, 11, 7):
            raise RuntimeError("Door observation/action dimensions changed")
        if demos.data["robot"].shape[1:] != (robot_dim,):
            raise RuntimeError("Expert/live observation mismatch")
        expert_fraction, do_bc, online_lr = 0.5, True, 3e-4
        bc_weight = BC_WEIGHTS[args.variant]
        recipe = dict(expert_fraction=expert_fraction, bc_initialized=do_bc,
                      online_learning_rate=online_lr, bc_weight=bc_weight,
                      offline_critic_updates=0, fixed_alpha_schedule=args.variant,
                      rollout_std_cap=None, guard_threshold=None,
                      guard_patience=None)
        agent = Phase6SAC(robot_dim, gt_dim, action_dim, env.device, online_lr)
        online = ReplayBuffer(args.steps + args.num_envs, robot_dim, gt_dim,
                              action_dim, device=env.device)
        replay = ExpertOnlineReplay(demos, online, args.batch_size, expert_fraction)
        torch.manual_seed(args.seed + 100000)
        if do_bc:
            for update in range(args.bc_updates):
                batch = demos.sample(args.batch_size)
                bc_loss = agent.bc_step(batch["robot"], batch["privileged"],
                                        batch["action"])
                if update % 100 == 0:
                    writer.add_scalar("bc/loss", bc_loss, update)
            agent.reset_optimizer_after_bc()
        agent.set_online_learning_rate()
        for group in agent.alpha_opt.param_groups:
            group["lr"] = 0.0
        with torch.no_grad():
            agent.log_alpha.fill_(math.log(fixed_alpha(args.variant, 0)))
        # Frozen expert transitions make all drift/Q readings comparable.
        torch.manual_seed(args.seed + 200000)
        reference = reference_snapshot(agent, demos.sample(2048))
        fixed_q_batch = demos.sample(2048)
        eval_client = EvalClient(ROOT, f"P7{args.variant}", args.seed,
                                 args.num_envs, device)
        guard = CheckpointGuard(threshold=0.25, patience=1) if args.variant == "G1" else None
        eval_csv = OUT / f"eval_{run}.csv"
        diag_csv = OUT / f"diagnostics_{run}.csv"
        eval_header = ("variant", "seed", "env_steps", "success_rate",
                       "mean_return", "contact_rate", "eval_episodes",
                       "eval_env_steps", "wall_time_s", "train_success_rate",
                       "train_episodes", "raw_success_rate", "guard_event",
                       "best_guard_success", "rollback_count", "fixed_alpha")
        diag_header = ("variant", "seed", "env_steps", "optimizer_steps",
                       "critic_loss", "actor_loss", "bc_loss", "q_mean",
                       "q_variance", "q_disagreement", "policy_entropy",
                       "alpha", "action_drift_mse", "expert_action_mse",
                       "train_success_rate", "train_episodes")
        success, mean_return, contact, episodes, eval_steps = eval_client.evaluate(
            agent, 50000 + args.seed, args.eval_rounds)
        append_csv(eval_csv, eval_header,
                   (args.variant, args.seed, 0, success, mean_return, contact,
                    episodes, eval_steps, time.perf_counter()-started, float("nan"), 0,
                    success, "initial", success, 0, fixed_alpha(args.variant, 0)))
        if guard is not None:
            guard.observe(agent, success, 0)
        save_checkpoint(agent, CHECKPOINTS / run / "step_0.pt", 0, recipe)
        save_checkpoint(agent, CHECKPOINTS / run / "best.pt", 0, recipe)
        best_observed_success = success
        writer.add_scalar("eval/success_rate", success, 0)
        print(f"{run}: initial success={success:.3f}, BC={do_bc}, expert fraction={expert_fraction}", flush=True)
        env.reset(seed=args.seed)
        steps = optimizer_steps = 0
        next_eval = args.eval_every
        finished_successes = deque(maxlen=512)
        interval_successes = []
        losses = []
        while steps < args.steps:
            robot = robot_observation(env).clone()
            gt = env.get_tool_state().clone()
            agent.target_entropy = -action_dim
            with torch.no_grad():
                agent.log_alpha.fill_(math.log(fixed_alpha(args.variant, steps)))
            action = sample_rollout_action(agent, robot, args.variant)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            online.add({"robot": robot, "privileged": gt,
                        "action": action.detach(), "reward": reward.reshape(-1, 1),
                        "next_robot": next_robot, "next_privileged": next_gt,
                        "done": done.float().reshape(-1, 1)})
            if done.any():
                outcomes = (env.final_door_angle[done, 0] > SUCCESS_ANGLE_RAD).float().tolist()
                interval_successes.extend(outcomes)
                finished_successes.extend(outcomes)
            steps += env.num_envs
            for _ in range(args.updates_per_vector_step):
                batch = replay.sample()
                expert_batch = demos.sample(args.batch_size) if bc_weight else None
                metrics = agent.online_update(batch, expert_batch, bc_weight)
                optimizer_steps += 1
                losses.append((metrics["critic_loss"], metrics["actor_loss"],
                               metrics["bc_loss"]))
            if steps >= next_eval or steps >= args.steps:
                success, mean_return, contact, episodes, extra = eval_client.evaluate(
                    agent, 50000 + args.seed, args.eval_rounds)
                eval_steps += extra
                raw_success = success
                guard_event = "none"
                if guard is not None:
                    guard_event = guard.observe(agent, success, steps)
                    if guard_event == "rollback":
                        success, mean_return, contact, episodes, extra = eval_client.evaluate(
                            agent, 50000 + args.seed, args.eval_rounds)
                        eval_steps += extra
                best_guard = guard.best_success if guard is not None else float("nan")
                rollback_count = guard.rollbacks if guard is not None else 0
                train_success = float(np.mean(interval_successes)) if interval_successes else float("nan")
                append_csv(eval_csv, eval_header,
                           (args.variant, args.seed, steps, success, mean_return,
                            contact, episodes, eval_steps, time.perf_counter()-started,
                            train_success, len(interval_successes), raw_success,
                            guard_event, best_guard, rollback_count,
                            fixed_alpha(args.variant, steps)))
                diagnostic = measure(agent, reference, fixed_q_batch)
                avg_losses = np.mean(np.asarray(losses), axis=0)
                append_csv(diag_csv, diag_header,
                           (args.variant, args.seed, steps, optimizer_steps,
                            *avg_losses, *(diagnostic[key] for key in (
                                "q_mean", "q_variance", "q_disagreement",
                                "policy_entropy", "alpha", "action_drift_mse",
                                "expert_action_mse")), train_success,
                            len(interval_successes)))
                for name, value in diagnostic.items():
                    writer.add_scalar(f"diagnostics/{name}", value, steps)
                for name, value in zip(("critic_loss", "actor_loss", "bc_loss"), avg_losses):
                    writer.add_scalar(f"sac/{name}", value, steps)
                writer.add_scalar("eval/success_rate", success, steps)
                writer.add_scalar("eval/raw_success_rate", raw_success, steps)
                writer.add_scalar("stability/fixed_alpha", fixed_alpha(args.variant, steps), steps)
                writer.add_scalar("stability/rollback_count", rollback_count, steps)
                writer.add_scalar("eval/mean_return", mean_return, steps)
                if np.isfinite(train_success):
                    writer.add_scalar("train/interval_success_rate", train_success, steps)
                save_checkpoint(agent, CHECKPOINTS / run / f"step_{steps}.pt", steps, recipe)
                if success > best_observed_success:
                    best_observed_success = success
                    save_checkpoint(agent, CHECKPOINTS / run / "best.pt", steps, recipe)
                print(f"{run}: steps={steps} success={success:.3f} raw={raw_success:.3f} guard={guard_event} train={train_success:.3f} drift={diagnostic['action_drift_mse']:.4f}", flush=True)
                next_eval += args.eval_every
                interval_successes.clear()
                losses.clear()
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
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"P7{args.variant}_seed{args.seed}.error.txt").write_text(detail)
        print(detail, flush=True)
        raise
    finally:
        app.close()
