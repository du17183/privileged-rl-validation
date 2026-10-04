"""Fixed-ratio expert/online transition replay for Phase 6.

The requested ratio is applied to every optimizer minibatch, not to a
concatenated buffer where different episode lengths would change it.
"""

import torch

from algorithms.replay import FIELDS


class ExpertOnlineReplay:
    def __init__(self, expert, online, batch_size: int, expert_fraction: float):
        if not 0.0 <= expert_fraction <= 1.0:
            raise ValueError("expert_fraction must be in [0, 1]")
        if batch_size < 1 or not len(expert):
            raise ValueError("expert data and a positive batch size are required")
        self.expert = expert
        self.online = online
        self.batch_size = batch_size
        self.expert_fraction = expert_fraction

    def sample(self):
        n_expert = round(self.batch_size * self.expert_fraction)
        n_online = self.batch_size - n_expert
        if n_online and not len(self.online):
            raise RuntimeError("Collect online transitions before replay updates")
        chunks = []
        if n_expert:
            chunks.append(self.expert.sample(n_expert))
        if n_online:
            chunks.append(self.online.sample(n_online))
        return {key: torch.cat([part[key] for part in chunks], dim=0) for key in FIELDS}
