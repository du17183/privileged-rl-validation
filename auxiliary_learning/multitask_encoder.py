"""Phase 4 policy supervision and robot-only shared value representation.

Fixture labels are consumed only by training losses. ``act`` needs robot
proprioception alone, including for the shared-value variant.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.asymmetric_sac import TwinQ
from auxiliary_learning.gt_prediction import AuxiliarySAC, EncodedGaussianActor


class MultiTaskActor(EncodedGaussianActor):
    """Separate angle/contact heads on the same robot-only policy encoder."""

    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.gt_head = nn.Identity()  # keep the policy/encoder initialization intact
        self.angle_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.contact_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))

    def predict_gt(self, obs):
        z = self.encoder(obs)
        return torch.cat((self.angle_head(z), self.contact_head(z)), dim=-1)


class MultiTaskSAC(AuxiliarySAC):
    """E100M: independent prediction heads, same 0.1*(angle+contact) loss."""

    def __init__(self, robot_dim, privileged_dim, action_dim, device="cuda", aux_weight=0.1):
        super().__init__(robot_dim, privileged_dim, action_dim, device=device, aux_weight=aux_weight)
        paired_actor = self.actor
        self.actor = MultiTaskActor(robot_dim, action_dim).to(self.device)
        self.actor.encoder.load_state_dict(paired_actor.encoder.state_dict())
        self.actor.policy_head.load_state_dict(paired_actor.policy_head.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)


class SharedEncoderSAC(AuxiliarySAC):
    """Robot-only Q head consumes policy-encoder features, with GT supervision.

    The critic optimizer updates Q heads only. Policy and auxiliary gradients
    update the encoder. A slowly updated target encoder prevents next-state
    targets from using the latest encoder weights immediately.
    """

    def __init__(self, robot_dim, privileged_dim, action_dim, device="cuda", aux_weight=0.0):
        super().__init__(robot_dim, privileged_dim, action_dim, device=device, aux_weight=aux_weight)
        self.critic = TwinQ(256, action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_encoder = copy.deepcopy(self.actor.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    def critic_input(self, robot, privileged=None):
        return self.actor.encoder(robot)

    def critic_step(self, batch):
        robot, nxt_robot = batch["robot"], batch["next_robot"]
        action, reward, done = batch["action"], batch["reward"], batch["done"]
        with torch.no_grad():
            z = self.actor.encoder(robot)
            next_z = self.target_encoder(nxt_robot)
            next_action, next_logp = self.actor(nxt_robot)
            tq1, tq2 = self.target_critic(next_z, next_action)
            target = reward + self.cfg.gamma * (1.0 - done) * (
                torch.min(tq1, tq2) - self.log_alpha.exp() * next_logp
            )
        q1, q2 = self.critic(z, action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.lerp_(source_param, self.cfg.tau)
            for target_param, source_param in zip(self.target_encoder.parameters(), self.actor.encoder.parameters()):
                target_param.lerp_(source_param, self.cfg.tau)
        return float(loss.detach())

    def update(self, batch, bc_batch=None, bc_weight=0.0):
        critic_loss = self.critic_step(batch)
        robot, gt = batch["robot"], batch["privileged"]
        for p in self.critic.parameters():
            p.requires_grad_(False)
        z = self.actor.encoder(robot)
        action, logp = self.actor(robot)
        q1, q2 = self.critic(z, action)
        actor_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1, q2)).mean()
        bc_loss = torch.zeros((), device=self.device)
        if bc_weight:
            bc_action, _ = self.actor(bc_batch["robot"], deterministic=True)
            bc_loss = F.mse_loss(bc_action, bc_batch["action"])
            actor_loss = actor_loss + bc_weight * bc_loss
        pred = self.actor.predict_gt(robot)
        angle_loss = F.mse_loss(pred[:, :1], gt[:, :1])
        contact_loss = F.binary_cross_entropy_with_logits(pred[:, 1:], gt[:, 9:11])
        actor_loss = actor_loss + self.aux_weight * (angle_loss + contact_loss)
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

    def checkpoint(self):
        state = super().checkpoint()
        state["target_encoder"] = self.target_encoder.state_dict()
        return state
