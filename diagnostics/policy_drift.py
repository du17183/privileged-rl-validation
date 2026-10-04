"""Fixed-expert policy drift and Q diagnostics, evaluated without updating nets."""

import torch


@torch.no_grad()
def reference_snapshot(agent, expert_batch):
    obs = expert_batch["robot"]
    action, _ = agent.actor(obs, deterministic=True)
    return {"observation": obs.detach().clone(),
            "bc_action": action.detach().clone(),
            "expert_action": expert_batch["action"].detach().clone()}


@torch.no_grad()
def measure(agent, reference, q_batch):
    action, _ = agent.actor(reference["observation"], deterministic=True)
    _, logp = agent.actor(reference["observation"])
    q1, q2 = agent.critic(q_batch["robot"], q_batch["action"])
    q = torch.minimum(q1, q2)
    return {
        "action_drift_mse": float((action-reference["bc_action"]).square().mean()),
        "expert_action_mse": float((action-reference["expert_action"]).square().mean()),
        "q_mean": float(q.mean()),
        "q_variance": float(q.var(unbiased=False)),
        "q_disagreement": float((q1-q2).abs().mean()),
        "policy_entropy": float(-logp.mean()),
        "alpha": float(agent.log_alpha.exp()),
    }
