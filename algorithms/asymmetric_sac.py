"""SAC with three precisely defined observation routes.

Actor input: robot observation for A/B; robot + privileged for C.
Critic input: robot observation for A; robot + privileged for B/C.
All actions are normalized to [-1, 1] before entering this module.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(in_dim, out_dim, hidden=256):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.act_dim = act_dim
        # BC supervises the mean only. Start online exploration near the
        # demonstrated action instead of using an untrained ~unit Gaussian.
        output = self.net[-1]
        with torch.no_grad():
            output.weight[act_dim:].zero_()
            output.bias[act_dim:].fill_(-2.0)

    def forward(self, obs, deterministic=False):
        mean, log_std = self.net(obs).split(self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1)
        self.q2 = mlp(obs_dim + act_dim, 1)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    bc_lr: float = 3e-4
    target_entropy_scale: float = 1.0
    initial_alpha: float = 0.05


class AsymmetricSAC:
    def __init__(self, robot_dim, privileged_dim, action_dim, variant, device="cuda", cfg=None):
        if variant not in ("A", "B", "C"):
            raise ValueError("variant must be A, B, or C")
        self.variant = variant
        self.device = torch.device(device)
        self.cfg = cfg or SACConfig()
        self.actor_dim = robot_dim + (privileged_dim if variant == "C" else 0)
        self.critic_dim = robot_dim + (privileged_dim if variant in ("B", "C") else 0)
        self.actor = GaussianActor(self.actor_dim, action_dim).to(self.device)
        self.critic = TwinQ(self.critic_dim, action_dim).to(self.device)
        self.target_critic = TwinQ(self.critic_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.log_alpha = torch.tensor(
            math.log(self.cfg.initial_alpha), device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        self.target_entropy = -action_dim * self.cfg.target_entropy_scale

    def actor_input(self, robot, privileged=None):
        if self.variant == "C":
            if privileged is None:
                raise ValueError("Privileged policy requires GT at inference")
            return torch.cat((robot, privileged), dim=-1)
        return robot

    def critic_input(self, robot, privileged=None):
        if self.variant in ("B", "C"):
            if privileged is None:
                raise ValueError("Privileged critic requires GT during training")
            return torch.cat((robot, privileged), dim=-1)
        return robot

    @torch.no_grad()
    def act(self, robot, privileged=None, deterministic=False):
        """B deliberately ignores privileged, including at evaluation time."""
        return self.actor(self.actor_input(robot, privileged), deterministic)[0]

    def bc_step(self, robot, privileged, expert_action):
        pred, _ = self.actor(self.actor_input(robot, privileged), deterministic=True)
        loss = F.mse_loss(pred, expert_action)
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return float(loss.detach())

    def critic_step(self, batch):
        """Fit Q from replay while leaving the BC-initialized actor untouched."""
        robot, gt = batch["robot"], batch["privileged"]
        nxt_robot, nxt_gt = batch["next_robot"], batch["next_privileged"]
        action, reward, done = batch["action"], batch["reward"], batch["done"]
        critic_obs = self.critic_input(robot, gt)
        with torch.no_grad():
            next_action, next_logp = self.actor(self.actor_input(nxt_robot, nxt_gt))
            q1_target, q2_target = self.target_critic(self.critic_input(nxt_robot, nxt_gt), next_action)
            target = reward + self.cfg.gamma * (1.0 - done) * (
                torch.min(q1_target, q2_target) - self.log_alpha.exp() * next_logp
            )
        q1, q2 = self.critic(critic_obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.lerp_(source_param, self.cfg.tau)
        return float(critic_loss.detach())

    def update(self, batch, bc_batch=None, bc_weight=0.0):
        """SAC update, optionally retaining demo behavior during early online RL."""
        critic_loss = self.critic_step(batch)
        robot, gt = batch["robot"], batch["privileged"]
        critic_obs = self.critic_input(robot, gt)

        # Critic parameters are frozen for the actor step, but its derivative wrt action is retained.
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_action, logp = self.actor(self.actor_input(robot, gt))
        q1_new, q2_new = self.critic(critic_obs, new_action)
        actor_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1_new, q2_new)).mean()
        bc_loss = torch.zeros((), device=self.device)
        if bc_weight > 0:
            if bc_batch is None:
                raise ValueError("bc_batch is required for BC-regularized SAC")
            bc_action, _ = self.actor(
                self.actor_input(bc_batch["robot"], bc_batch["privileged"]), deterministic=True
            )
            bc_loss = F.mse_loss(bc_action, bc_batch["action"])
            actor_loss = actor_loss + bc_weight * bc_loss
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        return {
            "critic_loss": critic_loss,
            "actor_loss": float(actor_loss.detach()),
            "bc_loss": float(bc_loss.detach()),
            "bc_weight": bc_weight,
            "alpha": float(self.log_alpha.exp().detach()),
        }

    def checkpoint(self):
        return {
            "variant": self.variant,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
        }
