"""Deterministic validation monitor and complete SAC-state rollback."""

import copy
from dataclasses import dataclass

import torch


def capture_agent(agent):
    state = {"actor": copy.deepcopy(agent.actor.state_dict()),
             "critic": copy.deepcopy(agent.critic.state_dict()),
             "target_critic": copy.deepcopy(agent.target_critic.state_dict()),
             "actor_opt": copy.deepcopy(agent.actor_opt.state_dict()),
             "critic_opt": copy.deepcopy(agent.critic_opt.state_dict()),
             "alpha_opt": copy.deepcopy(agent.alpha_opt.state_dict()),
             "log_alpha": agent.log_alpha.detach().clone()}
    if hasattr(agent, "target_encoder"):
        state["target_encoder"] = copy.deepcopy(agent.target_encoder.state_dict())
    return state


def restore_agent(agent, state):
    for key in ("actor", "critic", "target_critic"):
        getattr(agent, key).load_state_dict(state[key])
    for key in ("actor_opt", "critic_opt", "alpha_opt"):
        getattr(agent, key).load_state_dict(state[key])
    with torch.no_grad():
        agent.log_alpha.copy_(state["log_alpha"])
    if "target_encoder" in state:
        agent.target_encoder.load_state_dict(state["target_encoder"])


@dataclass
class RollbackDecision:
    improved: bool
    rollback: bool
    best_success: float
    bad_evaluations: int


class RollbackMonitor:
    """Two consecutive drops avoid reacting to one noisy 32-episode score.

    Replay is intentionally retained across rollback; model and optimizers
    return to the best snapshot. This is a training intervention, distinct
    from selecting the best model only at deployment time.
    """

    def __init__(self, threshold=0.25, min_peak=0.5, patience=2):
        if not 0 < threshold <= 1 or patience < 1:
            raise ValueError("Invalid rollback threshold or patience")
        self.threshold = threshold
        self.min_peak = min_peak
        self.patience = patience
        self.best_success = float("-inf")
        self.best_state = None
        self.bad_evaluations = 0
        self.rollback_count = 0

    def observe(self, agent, success):
        improved = success > self.best_success
        if improved:
            self.best_success = success
            self.best_state = capture_agent(agent)
            self.bad_evaluations = 0
        elif self.best_success >= self.min_peak and success < self.best_success - self.threshold:
            self.bad_evaluations += 1
        else:
            self.bad_evaluations = 0
        rollback = self.bad_evaluations >= self.patience
        if rollback:
            restore_agent(agent, self.best_state)
            self.bad_evaluations = 0
            self.rollback_count += 1
        return RollbackDecision(improved, rollback, self.best_success, self.bad_evaluations)
