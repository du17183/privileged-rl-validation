"""Measured-input evaluator with per-episode physical condition records."""
import math
import torch
from door_env.door import robot_observation
from progress_rl.progress_monitor import EpisodeMonitor, aggregate
from safe_online.kl_constraint import normal_kl
from environment_feedback.normalization import encode
from experiments.phase12_environment_state_feedback.data import transition
from experiments.phase12_environment_state_feedback.conditions import CONDITIONS
from evaluation.stratified_generalization import slice_episodes


@torch.no_grad()
def evaluate(actor, anchor, env, arm, center, seed, condition='level2', mode='policy', rounds=2,
             ablation=None, noise=None):
    if condition.startswith('curriculum'):
        level, preset = int(condition[-1]), None
    else:
        level, preset = CONDITIONS[condition]
    env.set_randomization(level, preset)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    lengths = torch.zeros_like(counts)
    returns = torch.zeros(env.num_envs, device=env.device)
    monitor = EpisodeMonitor(env.num_envs, env.device)
    initial = env.parameters.clone()
    initial_handle = env.get_environment_state()['handle_position'].clone()
    records = []
    totals = torch.zeros(5, device=env.device)
    samples = torch.zeros((), device=env.device)
    interactions = 0
    for _ in range((env.max_episode_length+2)*(rounds+1)):
        state = {k: v.clone() for k, v in env.get_environment_state().items()}
        if noise is not None:
            state = noise(state)
            state['progress'] = ((state['door_angle']-env.episode_start)/(state['target_angle']-env.episode_start).clamp_min(1e-6)).clamp(0, 1)
        if ablation:
            from environment_feedback.normalization import mask_state
            state = mask_state(state, ablation, center)
        observation = encode(robot_observation(env), state, arm, center)
        mean, log_std = actor.distribution(observation)
        old_mean, old_std = anchor.distribution(observation)
        action = mean.tanh() if mode == 'deterministic' else (mean+log_std.exp()*torch.randn_like(mean)).tanh()
        active = (counts < rounds).float()
        values = torch.stack((log_std.exp().mean(-1),
            normal_kl(mean, log_std, old_mean, old_std), (mean.tanh()-old_mean.tanh()).square().mean(-1),
            (log_std+.5*math.log(2*math.pi*math.e)).sum(-1), mean.tanh().abs().mean(-1)), -1)
        totals += (values*active[:, None]).sum(0)
        samples += active.sum()
        before = env.get_tool_state()[:, :1].clone()
        _, reward, terminated, truncated, _ = env.step(action)
        _, gt, done = transition(env, terminated, truncated, arm, center)
        interactions += env.num_envs
        lengths += 1
        returns += reward
        monitor.update(before, gt[:, :1], (gt[:, 9:11]>.5).all(-1, keepdim=True))
        for i in done.nonzero().squeeze(-1).tolist():
            row = monitor.finish(i, gt[i, 0], float(env.final_target[i, 0]), lengths[i], float(env.final_start[i, 0]))
            p = initial[i]
            row.update(return_value=float(returns[i]), initial_angle_deg=math.degrees(float(p[0])),
                offset_x_m=float(p[1]), offset_y_m=float(p[2]), offset_z_m=float(p[3]),
                offset_linf_cm=100*float(p[1:4].abs().max()), friction_scale=float(p[4]),
                initial_handle_position=initial_handle[i].tolist(),
                final_handle_position=env.final_environment['handle_position'][i].tolist(),
                env_index=i, episode_index=int(counts[i]))
            if counts[i] < rounds:
                records.append(row)
                counts[i] += 1
            lengths[i] = 0
            returns[i] = 0
            initial[i] = env.parameters[i]
            initial_handle[i] = env.get_environment_state()['handle_position'][i]
        if bool((counts >= rounds).all()): break
    if len(records) != env.num_envs*rounds: raise RuntimeError('Incomplete physical evaluation')
    scalar_keys = [k for k, v in records[0].items() if isinstance(v, (float, int))]
    result = aggregate([{k: r[k] for k in scalar_keys} for r in records])
    result.update(episodes=len(records), eval_env_steps=interactions, mode=mode, condition=condition,
                  physical_reset_checks=env.physical_reset_checks)
    for i, key in enumerate(('effective_std', 'anchor_kl', 'anchor_action_mse', 'pre_tanh_entropy', 'mean_action_abs')):
        result[key] = float(totals[i]/samples)
    result['slices'] = slice_episodes(records)
    return dict(metrics=result, records=records)
