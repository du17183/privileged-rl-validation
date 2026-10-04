"""Robot-only policy encoder with training-only angle/contact prediction.

The prediction head is never called by the deployment policy. Both E0 and
E1 use this exact architecture; E0 simply gives the auxiliary loss weight 0.
"""

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.asymmetric_sac import AsymmetricSAC, TwinQ


class EncodedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU())
        self.policy_head = nn.Linear(256, 2 * act_dim)
        self.gt_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.act_dim = act_dim
        with torch.no_grad():
            self.policy_head.weight[act_dim:].zero_()
            self.policy_head.bias[act_dim:].fill_(-2.0)

    def forward(self, obs, deterministic=False):
        mean, log_std = self.policy_head(self.encoder(obs)).split(self.act_dim, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        if deterministic:
            return torch.tanh(mean), None
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = torch.tanh(raw)
        logp = normal.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        return action, logp.sum(dim=-1, keepdim=True)

    def predict_gt(self, obs):
        return self.gt_head(self.encoder(obs))


class AuxiliarySAC(AsymmetricSAC):
    """Robot-only SAC with a policy encoder shared by policy and GT head."""

    def __init__(self, robot_dim, privileged_dim, action_dim, device="cuda", aux_weight=0.0):
        super().__init__(robot_dim, privileged_dim, action_dim, "A", device=device)
        self.actor = EncodedGaussianActor(robot_dim, action_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.aux_weight = float(aux_weight)

    def bc_step_with_gt(self, robot, privileged, expert_action, aux_weight=0.1):
        """Same BC minibatch/update count with GT supervision during BC only."""
        pred_action, _ = self.actor(robot, deterministic=True)
        bc_loss = F.mse_loss(pred_action, expert_action)
        pred_gt = self.actor.predict_gt(robot)
        aux_loss = F.mse_loss(pred_gt[:, :1], privileged[:, :1]) + \
            F.binary_cross_entropy_with_logits(pred_gt[:, 1:], privileged[:, 9:11])
        loss = bc_loss + aux_weight * aux_loss
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return float(bc_loss.detach()), float(aux_loss.detach())

    def update(self, batch, bc_batch=None, bc_weight=0.0):
        # The actor update below is identical to A except for the optional
        # supervised gradient through the policy encoder on the same batch.
        critic_loss = self.critic_step(batch)
        robot, gt = batch["robot"], batch["privileged"]
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_action, logp = self.actor(robot)
        q1, q2 = self.critic(robot, new_action)
        actor_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1, q2)).mean()
        bc_loss = torch.zeros((), device=self.device)
        if bc_weight:
            bc_action, _ = self.actor(bc_batch["robot"], deterministic=True)
            bc_loss = F.mse_loss(bc_action, bc_batch["action"])
            actor_loss = actor_loss + bc_weight * bc_loss
        pred = self.actor.predict_gt(robot)
        angle_loss = F.mse_loss(pred[:, :1], gt[:, :1])
        contact_loss = F.binary_cross_entropy_with_logits(pred[:, 1:], gt[:, 9:11])
        aux_loss = angle_loss + contact_loss
        actor_loss = actor_loss + self.aux_weight * aux_loss
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        return {"critic_loss": critic_loss, "actor_loss": float(actor_loss.detach()),
                "bc_loss": float(bc_loss.detach()), "bc_weight": bc_weight,
                "alpha": float(self.log_alpha.exp().detach()),
                "aux_angle_loss": float(angle_loss.detach()),
                "aux_contact_loss": float(contact_loss.detach())}
