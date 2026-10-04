"""Online trajectory-value replay with a uniform support floor.

The current privileged critic assigns a *predicted* mean Q to each completed
online trajectory. Success is logged but never enters the priority formula.
Within a selected trajectory, transitions are sampled uniformly. The 20%
uniform mixture keeps failed and recent experience available to SAC.
"""

from collections import deque
import math
import statistics
import torch

from algorithms.replay import ReplayBuffer


class ValueWeightedReplay(ReplayBuffer):
    def __init__(self, capacity, robot_dim, privileged_dim, action_dim, device="cuda",
                 uniform_fraction=0.2, temperature=1.0):
        super().__init__(capacity, robot_dim, privileged_dim, action_dim, device)
        if not 0 < uniform_fraction <= 1 or temperature <= 0:
            raise ValueError("uniform_fraction must be in (0,1], temperature > 0")
        self.uniform_fraction = uniform_fraction
        self.temperature = temperature
        self.episode_id = torch.full((capacity,), -1, device=self.device, dtype=torch.long)
        self.value = torch.zeros(capacity, device=self.device)
        self.success = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.priority = torch.full((capacity,), 1 / 300, device=self.device)
        self._scores = deque(maxlen=256)
        self._cdf = None
        self._distribution = None
        self._dirty = True
        self._adds_since_refresh = 0
        self.finalized = 0

    def add(self, batch, episode_ids):
        count = len(batch["action"])
        if count > self.capacity:
            raise ValueError("A vector batch cannot exceed replay capacity")
        ids = torch.as_tensor(episode_ids, device=self.device, dtype=torch.long)
        if ids.shape != (count,):
            raise ValueError("episode_ids must match vector batch")
        indices = (torch.arange(count, device=self.device) + self.cursor) % self.capacity
        super().add(batch)
        self.episode_id[indices] = ids
        self.value[indices] = 0
        self.success[indices] = False
        self.priority[indices] = 1 / 300
        self._adds_since_refresh += 1
        if self._adds_since_refresh >= 32:
            self._dirty = True
            self._adds_since_refresh = 0
        return indices.tolist()

    @torch.no_grad()
    def finalize(self, episode_id, indices, critic, succeeded):
        if not indices:
            return None
        idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        idx = idx[self.episode_id[idx] == episode_id]
        if not len(idx):
            return None
        obs = torch.cat((self.data["robot"][idx], self.data["privileged"][idx]), dim=-1)
        q1, q2 = critic(obs, self.data["action"][idx])
        score = float(torch.min(q1, q2).mean())
        if len(self._scores) >= 16:
            centre = statistics.mean(self._scores)
            spread = max(statistics.pstdev(self._scores), 0.25)
            normalized = max(-2.0, min(2.0, (score - centre) / spread))
        else:
            normalized = 0.0
        self._scores.append(score)
        trajectory_weight = math.exp(normalized / self.temperature)
        self.value[idx] = score
        self.success[idx] = bool(succeeded)
        self.priority[idx] = trajectory_weight / len(idx)
        self.finalized += 1
        self._dirty = True
        return score

    def _refresh(self):
        if not self._dirty and self._cdf is not None:
            return
        priority = self.priority[:self.size]
        weighted = priority / priority.sum()
        probabilities = self.uniform_fraction / self.size + (1 - self.uniform_fraction) * weighted
        self._distribution = probabilities
        self._cdf = torch.cumsum(probabilities, dim=0)
        self._cdf[-1] = 1.0
        self._dirty = False

    def sample(self, count):
        if not self.size:
            raise ValueError("Replay buffer is empty")
        self._refresh()
        idx = torch.searchsorted(self._cdf, torch.rand(count, device=self.device)).clamp(max=self.size - 1)
        return {key: value[idx] for key, value in self.data.items()}

    @torch.no_grad()
    def sampling_stats(self):
        if not self.size:
            return {"effective_sample_size": 0., "top_decile_mass": 0.,
                    "success_sampling_mass": 0., "success_uniform_fraction": 0.,
                    "completed_trajectories": self.finalized}
        self._dirty = True
        self._refresh()
        p = self._distribution
        top = max(1, math.ceil(self.size / 10))
        return {
            "effective_sample_size": float(1 / p.square().sum()),
            "top_decile_mass": float(torch.topk(p, top).values.sum()),
            "success_sampling_mass": float(p[self.success[:self.size]].sum()),
            "success_uniform_fraction": float(self.success[:self.size].float().mean()),
            "completed_trajectories": self.finalized,
        }

    def sample_experience(self, count):
        """Future DIVL/QAM boundary: state, action, reward, next_state, value, success."""
        if not self.size:
            raise ValueError("Replay buffer is empty")
        self._refresh()
        idx = torch.searchsorted(self._cdf, torch.rand(count, device=self.device)).clamp(max=self.size - 1)
        return {
            "state": {"robot": self.data["robot"][idx], "privileged": self.data["privileged"][idx]},
            "action": self.data["action"][idx], "reward": self.data["reward"][idx],
            "next_state": {"robot": self.data["next_robot"][idx],
                           "privileged": self.data["next_privileged"][idx]},
            "value": self.value[idx, None], "success": self.success[idx, None],
        }
