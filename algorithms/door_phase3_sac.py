"""Critic masks, actor-only freeze, and distilled robot-only switch."""

import copy
import torch
from torch.nn import functional as F

from algorithms.asymmetric_sac import AsymmetricSAC, TwinQ


class MaskedSAC(AsymmetricSAC):
    def __init__(self, robot_dim, privileged_dim, action_dim, indices, device="cuda"):
        self.gt_indices = tuple(indices)
        super().__init__(robot_dim, len(self.gt_indices), action_dim, "B", device=device)
        self.robot_dim = robot_dim
        self.action_dim = action_dim
        self.frozen = False
        self.switched = False

    def critic_input(self, robot, privileged=None):
        if self.switched:
            return robot
        if privileged is None:
            raise ValueError("GT required by privileged critic during training")
        return torch.cat((robot, privileged[:, self.gt_indices]), dim=-1)

    def freeze_critic(self):
        self.frozen = True
        self.critic.eval()
        self.target_critic.eval()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    def switch_to_robot_critic(self, replay, demos, batch_size=256, distill_updates=1000):
        """Fit a robot critic to the full-GT teacher before switching.

        Distillation uses existing replay only; it consumes zero new environment
        interactions. It is logged as extra optimizer steps in the experiment.
        """
        if self.frozen or self.switched:
            raise RuntimeError("Critic can be switched exactly once before freeze")
        teacher = self.critic
        student = TwinQ(self.robot_dim, self.action_dim).to(self.device)
        opt = torch.optim.Adam(student.parameters(), lr=self.cfg.lr)
        from algorithms.replay import mixed_sample
        for _ in range(distill_updates):
            batch = mixed_sample(replay, demos, batch_size, 0.25)
            with torch.no_grad():
                old_q1, old_q2 = teacher(self.critic_input(batch["robot"], batch["privileged"]),
                                         batch["action"])
            q1, q2 = student(batch["robot"], batch["action"])
            loss = F.mse_loss(q1, old_q1) + F.mse_loss(q2, old_q2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self.critic = student
        self.target_critic = copy.deepcopy(student)
        self.critic_opt = opt
        self.switched = True
        self.variant = "A"
        return float(loss.detach())

    def update(self, batch, bc_batch=None, bc_weight=0.0):
        if not self.frozen:
            return super().update(batch, bc_batch, bc_weight)
        # A frozen critic still differentiates through its action input.
        robot, gt = batch["robot"], batch["privileged"]
        action, logp = self.actor(robot)
        q1, q2 = self.critic(self.critic_input(robot, gt), action)
        actor_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1, q2)).mean()
        bc_loss = torch.zeros((), device=self.device)
        if bc_weight:
            bc_action, _ = self.actor(bc_batch["robot"], deterministic=True)
            bc_loss = F.mse_loss(bc_action, bc_batch["action"])
            actor_loss = actor_loss + bc_weight * bc_loss
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        return {"critic_loss": float("nan"), "actor_loss": float(actor_loss.detach()),
                "bc_loss": float(bc_loss.detach()), "bc_weight": bc_weight,
                "alpha": float(self.log_alpha.exp().detach())}

    def checkpoint(self):
        state = super().checkpoint()
        state.update({"gt_indices": self.gt_indices, "frozen": self.frozen,
                      "switched": self.switched})
        return state
