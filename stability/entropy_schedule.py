"""Phase 7 entropy target and rollout-only action variance controls."""

import torch


def fixed_alpha(variant: str, steps: int) -> float:
    """Direct entropy-coefficient intervention; automatic tuning is disabled."""
    if variant == "E1L":
        return 0.001
    if variant == "E2L":
        progress = min(max(steps, 0) / 200000.0, 1.0)
        return 0.05 + (0.001 - 0.05) * progress
    raise ValueError(f"No fixed-alpha schedule for {variant}")


def entropy_scale(variant: str, steps: int) -> float:
    """Historical exploratory target-magnitude setting, not low alpha.

    Kept only so the originally launched E1/E2 runs remain reproducible.
    New low-coefficient trials use fixed_alpha instead.
    """
    if variant == "E1":
        return 0.2
    if variant == "E2":
        return 1.0 - 0.8 * min(max(steps, 0) / 200000.0, 1.0)
    return 1.0


def sample_rollout_action(agent, observation, variant: str):
    if variant != "E3":
        return agent.act(observation)
    # This only changes the data collection policy, not SAC's training loss.
    with torch.no_grad():
        mean, log_std = agent.actor.policy_head(agent.actor.encoder(observation)).split(
            agent.actor.act_dim, dim=-1
        )
        std = log_std.clamp(-5.0, 2.0).exp().clamp(max=0.05)
        return torch.tanh(mean + std * torch.randn_like(mean))
