"""Robot-observation SAC with optional fixed expert behavior regularization."""

import torch

from auxiliary_learning.gt_prediction import AuxiliarySAC


class Phase6SAC(AuxiliarySAC):
    def __init__(self, robot_dim, privileged_dim, action_dim, device,
                 rl_learning_rate=3e-4):
        super().__init__(robot_dim, privileged_dim, action_dim, device=device,
                         aux_weight=0.0)
        self.rl_learning_rate = float(rl_learning_rate)

    def reset_optimizer_after_bc(self):
        """Discard BC Adam moments so online differences are policy weights only."""
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)

    def set_online_learning_rate(self):
        """Apply lower LR after identical BC pretraining, when requested."""
        for optimizer in (self.actor_opt, self.critic_opt, self.alpha_opt):
            for group in optimizer.param_groups:
                group["lr"] = self.rl_learning_rate

    def online_update(self, replay_batch, expert_batch=None, bc_weight=0.0):
        if bc_weight and expert_batch is None:
            raise ValueError("BC regularization requires an expert minibatch")
        return self.update(replay_batch, bc_batch=expert_batch,
                           bc_weight=bc_weight)
