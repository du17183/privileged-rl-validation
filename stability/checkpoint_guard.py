"""Evaluation-gated recovery of the complete SAC optimizer/model state."""

import copy

import torch


class CheckpointGuard:
    def __init__(self, threshold=0.25, patience=2):
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.best_success = -1.0
        self.best_step = 0
        self.best_state = None
        self.bad_count = 0
        self.rollbacks = 0

    def observe(self, agent, success, step):
        if success > self.best_success:
            self.best_success = float(success)
            self.best_step = int(step)
            self.best_state = copy.deepcopy(agent.checkpoint())
            self.bad_count = 0
            return "new_best"
        if success < self.best_success - self.threshold:
            self.bad_count += 1
        else:
            self.bad_count = 0
        if self.bad_count < self.patience:
            return "keep"
        self.restore(agent)
        self.bad_count = 0
        self.rollbacks += 1
        return "rollback"

    def restore(self, agent):
        if self.best_state is None:
            raise RuntimeError("No previously evaluated checkpoint available")
        state = self.best_state
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        agent.target_critic.load_state_dict(state["target_critic"])
        agent.actor_opt.load_state_dict(state["actor_opt"])
        agent.critic_opt.load_state_dict(state["critic_opt"])
        agent.alpha_opt.load_state_dict(state["alpha_opt"])
        with torch.no_grad():
            agent.log_alpha.copy_(state["log_alpha"])
