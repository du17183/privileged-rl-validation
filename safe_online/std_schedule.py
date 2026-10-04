"""The same executed distribution is used in actor loss and target-Q sampling."""
import math
import torch
from auxiliary_learning.gt_prediction import EncodedGaussianActor


class StdSchedule:
    def __init__(self, mode="original", upper=.01, start=.1, duration=100000):
        self.mode, self.upper, self.start, self.duration = mode, upper, start, duration

    def cap(self, steps):
        if self.mode == "original":
            return None
        if self.mode == "bound":
            return self.upper
        if self.mode != "anneal":
            raise ValueError(self.mode)
        # Apply a linear schedule on recorded 10k boundaries; compare incumbent
        # and candidate under the same current cap, never across two caps.
        t = min(1.0, (int(steps)//10000)*10000/self.duration)
        return self.start+t*(self.upper-self.start)


class ControlledActor(EncodedGaussianActor):
    """Same parameter names as Phase 8, plus an explicit runtime sigma cap."""
    def __init__(self, obs_dim=26, act_dim=7):
        super().__init__(obs_dim, act_dim)
        self.cap = None

    def distribution(self, observation, effective=True):
        mean, log_std = self.policy_head(self.encoder(observation)).split(self.act_dim, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        if effective and self.cap is not None:
            log_std = log_std.clamp_max(math.log(self.cap))
        return mean, log_std

    def forward(self, observation, deterministic=False):
        mean, log_std = self.distribution(observation)
        if deterministic:
            return torch.tanh(mean), None
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = torch.tanh(raw)
        # Preserve the historical SAC correction to isolate requested factors.
        logp = normal.log_prob(raw)-torch.log(1.0-action.square()+1e-6)
        return action, logp.sum(-1, keepdim=True)

