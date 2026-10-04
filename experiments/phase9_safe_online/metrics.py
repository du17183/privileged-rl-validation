"""Evaluate the actual learned sampling policy, independently of the learner."""
import math
import torch
from door_env.door import door_angle, robot_observation
from door_env.isaac_env import transition_after_step
from progress_rl.progress_monitor import EpisodeMonitor, aggregate
from safe_online.kl_constraint import normal_kl


@torch.no_grad()
def evaluate(actor, anchor, env, seed, mode="deterministic", rounds=2, trajectory_writer=None):
    torch.manual_seed(seed)
    env.reset(seed=seed)
    anchor.cap = actor.cap
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    lengths = torch.zeros_like(counts)
    returns = torch.zeros(env.num_envs, device=env.device)
    monitor = EpisodeMonitor(env.num_envs, env.device)
    density = torch.zeros(6, device=env.device)
    density_count = torch.zeros((), device=env.device)
    diagnostic_rng = torch.Generator(device=env.device).manual_seed(seed+15485863)
    records, history = [], []
    starts = [0]*env.num_envs
    interactions = 0
    for tick in range((env.max_episode_length+2)*(rounds+1)):
        obs = robot_observation(env).clone()
        gt = env.get_tool_state().clone()
        before = gt[:, :1]
        mean, log_std = actor.distribution(obs)
        raw_mean, raw_log_std = actor.distribution(obs, effective=False)
        anchor_mean, anchor_log_std = anchor.distribution(obs)
        old_mean, old_log_std = anchor.distribution(obs, effective=False)
        sigma = log_std.exp()
        if mode == "deterministic":
            action = mean.tanh()
        elif mode == "policy":
            action = (mean+sigma*torch.randn_like(mean)).tanh()
        elif mode == "noise001":
            action = (mean+.01*torch.randn_like(mean)).tanh()
        else:
            raise ValueError(mode)
        samples = (mean[None]+sigma[None]*torch.randn((8, *mean.shape), device=env.device,
                    generator=diagnostic_rng)).tanh()
        action_std = samples.std(0, unbiased=False)
        flip = torch.special.ndtr(-mean[:, -1].abs()/sigma[:, -1])
        values = torch.stack((sigma.mean(-1), raw_log_std.exp().mean(-1), action_std.mean(-1),
            action_std[:, :3].mean(-1), normal_kl(mean, log_std, anchor_mean, anchor_log_std),
            normal_kl(raw_mean, raw_log_std, old_mean, old_log_std)), dim=-1)
        active = (counts < rounds).float()
        density += (values*active[:, None]).sum(0)
        density_count += active.sum()
        _, reward, terminated, truncated, _ = env.step(action)
        next_obs, next_gt, done = transition_after_step(env, terminated, truncated)
        interactions += env.num_envs
        lengths += 1
        returns += reward
        monitor.update(before, next_gt[:, :1], (next_gt[:, 9:11]>.5).all(-1, keepdim=True))
        if trajectory_writer is not None:
            history.append(dict(robot=obs, privileged=gt, action=action.detach().clone(), reward=reward[:, None].clone(),
                next_robot=next_obs.clone(), next_privileged=next_gt.clone(), done=done.float()[:, None].clone()))
        for i in done.nonzero().squeeze(-1).tolist():
            row = monitor.finish(i, next_gt[i, 0], float(env.final_target[i, 0]), lengths[i],
                                 float(env.final_start[i, 0]))
            row["return"] = float(returns[i])
            if counts[i] < rounds:
                records.append(row)
                if trajectory_writer is not None:
                    trajectory = {key: torch.stack([history[t][key][i] for t in range(starts[i], tick+1)]).cpu().numpy()
                                  for key in history[0]}
                    trajectory_writer(trajectory, row)
                counts[i] += 1
            starts[i] = tick+1
            lengths[i] = 0
            returns[i] = 0
        if bool((counts >= rounds).all()):
            break
    if len(records) != env.num_envs*rounds:
        raise RuntimeError("Incomplete evaluation")
    result = dict(**aggregate(records), episodes=len(records), eval_env_steps=interactions, mode=mode)
    for i, key in enumerate(("effective_std", "raw_std", "action_std_mc", "xyz_action_std_mc", "anchor_kl", "raw_anchor_kl")):
        result[key] = float(density[i]/density_count)
    return result

