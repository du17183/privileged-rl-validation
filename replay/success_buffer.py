"""Explicit expert / frozen-anchor-success / fresh-exploration replay sources."""
import torch
from algorithms.replay import FIELDS


class SuccessReplay:
    def __init__(self, expert, online, anchor_success=None, expert_fraction=.5, batch_size=256):
        self.expert, self.online, self.anchor = expert, online, anchor_success
        self.expert_fraction, self.batch_size = expert_fraction, batch_size
        if not 0 <= expert_fraction <= 1:
            raise ValueError("Invalid expert fraction")
        self.draws = dict(expert=0, anchor_success=0, online_exploration=0)

    def sample(self):
        n_expert = round(self.batch_size*self.expert_fraction)
        remaining = self.batch_size-n_expert
        # Nonexpert portion is equally split between frozen-anchor successes
        # and fresh online exploration; online buffer keeps failures as well.
        n_anchor = remaining//2 if self.anchor is not None else 0
        counts = dict(expert=n_expert, anchor_success=n_anchor, online_exploration=remaining-n_anchor)
        pools = dict(expert=self.expert, anchor_success=self.anchor, online_exploration=self.online)
        chunks = []
        for source, count in counts.items():
            if count:
                if pools[source] is None or not len(pools[source]):
                    raise RuntimeError(f"Empty replay source {source}")
                chunks.append(pools[source].sample(count))
                self.draws[source] += count
        return {key: torch.cat([chunk[key] for chunk in chunks]) for key in FIELDS}

