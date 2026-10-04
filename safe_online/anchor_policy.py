"""Continue the full Phase 8 B checkpoint; never repeat BC initialization."""
import copy
import torch
from torch.nn import functional as F
from regularization.bc_regularized_sac import Phase6SAC
from safe_online.std_schedule import ControlledActor
from safe_online.kl_constraint import anchor_kl


class AnchoredSAC(Phase6SAC):
    def __init__(self, checkpoint, device, kl_weight=0.0):
        super().__init__(26, 11, 7, device)
        self.actor = ControlledActor().to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.kl_weight = float(kl_weight)
        self.load_state(checkpoint)
        self.anchor = copy.deepcopy(self.actor).eval()
        self.anchor.requires_grad_(False)

    def load_state(self, state):
        for key in ("actor", "critic", "target_critic", "actor_opt", "critic_opt", "alpha_opt"):
            getattr(self, key).load_state_dict(state[key])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.actor.cap = state.get("std_cap")
        self.target_entropy = state.get("target_entropy", -7.0)

    def checkpoint(self):
        state = super().checkpoint()
        state.update(std_cap=self.actor.cap, target_entropy=self.target_entropy,
                     kl_weight=self.kl_weight)
        return state

    def online_update(self, batch, expert_batch, bc_weight=10.0):
        critic_loss = self.critic_step(batch)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        action, logp = self.actor(batch["robot"])
        q1, q2 = self.critic(batch["robot"], action)
        rl_loss = (self.log_alpha.exp().detach()*logp-torch.minimum(q1, q2)).mean()
        bc_action, _ = self.actor(expert_batch["robot"], deterministic=True)
        bc_loss = F.mse_loss(bc_action, expert_batch["action"])
        # The anchor remains the same seed's Phase 8 actor throughout the run.
        kl = anchor_kl(self.actor, self.anchor, batch["robot"])
        actor_loss = rl_loss+bc_weight*bc_loss+self.kl_weight*kl
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        alpha_loss = -(self.log_alpha*(logp.detach()+self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        return dict(critic_loss=critic_loss, actor_loss=float(actor_loss.detach()),
                    bc_loss=float(bc_loss.detach()), anchor_kl=float(kl.detach()),
                    alpha=float(self.log_alpha.exp().detach()))

