"""Observable progress events and evaluation-gated complete-state recovery."""
import copy
import torch


class EpisodeMonitor:
    def __init__(self, count, device):
        self.maximum = torch.zeros(count, device=device)
        self.increase = torch.zeros_like(self.maximum)
        self.regression = torch.zeros_like(self.maximum)
        self.stall = torch.zeros_like(self.maximum, dtype=torch.long)
        self.max_stall = torch.zeros_like(self.stall)
        self.no_contact = torch.zeros_like(self.stall)
        self.max_no_contact = torch.zeros_like(self.stall)

    def update(self, before, after, contact):
        delta = (after-before).squeeze(-1)
        self.maximum = torch.maximum(self.maximum, after.squeeze(-1))
        self.increase += delta.clamp_min(0)
        self.regression += (-delta).clamp_min(0)
        self.stall = torch.where(delta < 1e-4, self.stall+1, 0)
        self.max_stall = torch.maximum(self.max_stall, self.stall)
        self.no_contact = torch.where(contact.squeeze(-1), 0, self.no_contact+1)
        self.max_no_contact = torch.maximum(self.max_no_contact, self.no_contact)

    def finish(self, i, angle, target, steps, start=0.0):
        value = float(angle)
        record = dict(max_angle=float(self.maximum[i]), final_angle=value,
                      cumulative_increase=float(self.increase[i]), regression_amount=float(self.regression[i]),
                      progress=min(1.0, max(0.0, (value-start)/(target-start))), success=float(value > target),
                      regression_event=float(self.regression[i] > 0.05),
                      stalled=float(self.max_stall[i] >= 120),
                      lost_contact=float(self.max_no_contact[i] >= 120), episode_steps=int(steps))
        for field in (self.maximum, self.increase, self.regression, self.stall, self.max_stall,
                      self.no_contact, self.max_no_contact):
            field[i] = 0
        return record


def aggregate(records):
    return {k: sum(r[k] for r in records)/len(records) for k in records[0]}


class ProgressGuard:
    def __init__(self):
        self.best = None
        self.state = None
        self.step = 0
        self.rollbacks = 0

    def observe(self, agent, metrics, step):
        if self.best is not None:
            decline = (metrics["success"] < self.best["success"]-0.25
                       or metrics["progress"] < self.best["progress"]-0.15
                       or metrics["max_angle"] < self.best["max_angle"]-0.15
                       or metrics["regression_amount"] > self.best["regression_amount"]+0.10)
            if decline:
                self.restore(agent)
                self.rollbacks += 1
                return "rollback"
            acceptable = (metrics["success"] >= self.best["success"]-0.05
                          and metrics["progress"] >= self.best["progress"]-0.05
                          and metrics["regression_amount"] <= self.best["regression_amount"]+0.05)
            better = (metrics["success"] > self.best["success"]+0.001
                      or (metrics["success"] >= self.best["success"]
                          and metrics["progress"] > self.best["progress"]+0.001))
            if not (acceptable and better):
                return "reject_best"
        self.best = dict(metrics)
        self.state = copy.deepcopy(agent.checkpoint())
        self.step = int(step)
        return "new_best"

    def restore(self, agent):
        for key in ("actor", "critic", "target_critic", "actor_opt", "critic_opt", "alpha_opt"):
            getattr(agent, key).load_state_dict(self.state[key])
        with torch.no_grad():
            agent.log_alpha.copy_(self.state["log_alpha"])
