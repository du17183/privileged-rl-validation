"""Fixed full-target evaluation using measurements, with separate action noise."""
import torch
from door_env.door import door_angle
from progress_rl.door_env import live_observation
from progress_rl.progress_monitor import EpisodeMonitor, aggregate


@torch.no_grad()
def evaluate(actor, env, variant, seed, rounds=2, noise=0.0):
    torch.manual_seed(seed)
    env.reset(seed=seed)
    monitor = EpisodeMonitor(env.num_envs, env.device)
    counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    length = torch.zeros_like(counts)
    records = []
    interactions = 0
    # A separate generator diagnoses the learned distribution without changing execution noise.
    diagnostic_rng = torch.Generator(device=env.device).manual_seed(seed+104729)
    density_sum = torch.zeros(4, device=env.device)
    contact_sum = torch.zeros_like(density_sum)
    density_count = torch.zeros((), device=env.device)
    contact_count = torch.zeros_like(density_count)
    for _ in range((env.max_episode_length+2)*(rounds+1)):
        before = door_angle(env).clone()
        obs = live_observation(env, variant)
        mean, log_std = actor.policy_head(actor.encoder(obs)).split(actor.act_dim, dim=-1)
        sigma = log_std.clamp(-5.0, 2.0).exp()
        action = torch.tanh(mean)
        samples = torch.tanh(mean[None]+sigma[None]*torch.randn((8, *mean.shape), device=env.device, generator=diagnostic_rng))
        actual_std = samples.std(dim=0, unbiased=False)
        flip = torch.special.ndtr(-mean[:, -1].abs()/sigma[:, -1])
        values = torch.stack((sigma.mean(-1), actual_std.mean(-1), actual_std[:, :3].mean(-1), flip), dim=-1)
        active = (counts < rounds).float()
        measured_contact = env.get_progress_state()["contact_state"].squeeze(-1)*active
        density_sum += (values*active[:, None]).sum(0)
        contact_sum += (values*measured_contact[:, None]).sum(0)
        density_count += active.sum()
        contact_count += measured_contact.sum()
        if noise:
            action = torch.tanh(torch.atanh(action.clamp(-0.999999, 0.999999))+noise*torch.randn_like(action))
        _, reward, terminated, truncated, _ = env.step(action)
        interactions += env.num_envs
        returns += reward
        length += 1
        done = terminated | truncated
        after = door_angle(env).clone()
        contact = env.get_progress_state()["contact_state"].bool()
        if done.any():
            after[done] = env.final_door_angle[done]
            contact[done] = env.final_progress[done, -1:].bool()
        monitor.update(before, after, contact)
        for i in done.nonzero().squeeze(-1).tolist():
            row = monitor.finish(i, after[i, 0], float(env.final_target[i, 0]), length[i],
                                 float(env.final_start[i, 0]))
            row["return"] = float(returns[i])
            if counts[i] < rounds:
                records.append(row)
                counts[i] += 1
            returns[i] = 0
            length[i] = 0
        if bool((counts >= rounds).all()):
            break
    if len(records) != env.num_envs*rounds:
        raise RuntimeError("Incomplete fixed evaluation")
    result = dict(**aggregate(records), episodes=len(records), eval_env_steps=interactions)
    for i, key in enumerate(("raw_policy_std", "action_std_mc", "xyz_action_std_mc", "gripper_sign_flip_probability")):
        result[f"mean_{key}"] = float(density_sum[i]/density_count)
        result[f"contact_{key}"] = float(contact_sum[i]/contact_count) if float(contact_count) else None
    result["contact_state_steps"] = int(contact_count)
    result["policy_state_steps"] = int(density_count)
    return result
