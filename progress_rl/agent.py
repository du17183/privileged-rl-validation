"""Matched BC+SAC initialization and a read-only expert replay transform."""
import torch
from regularization.bc_regularized_sac import Phase6SAC
from progress_rl.progress_state import observation
from progress_rl.progress_reward import relabel


def make_agent(variant, seed, device):
    torch.manual_seed(seed)
    base = Phase6SAC(26, 11, 7, device)
    if variant not in ("C", "D", "E"):
        return base
    agent = Phase6SAC(31, 11, 7, device)
    for module in ("actor", "critic", "target_critic"):
        source = getattr(base, module).state_dict()
        dest = getattr(agent, module).state_dict()
        for key, tensor in source.items():
            if tensor.shape == dest[key].shape:
                dest[key].copy_(tensor)
            else:
                assert tensor.ndim == 2 and dest[key].shape[1] == tensor.shape[1]+5
                dest[key].zero_()
                dest[key][:, :26] = tensor[:, :26]
                if module != "actor":
                    dest[key][:, 31:] = tensor[:, 26:]
        getattr(agent, module).load_state_dict(dest)
    del base
    return agent


class ExpertView:
    def __init__(self, source, variant, dt, target=1.0):
        self.source = source
        self.variant = variant
        self.dt = dt
        self.target = target

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        """Precompute the static transform rather than synchronizing each minibatch."""
        self._target = float(target)
        raw = {key: value[:len(self.source)] for key, value in self.source.data.items()}
        if self.variant == "E":
            # Keep each trajectory only through its first target-crossing transition.
            import numpy as np
            ends = np.flatnonzero(raw["done"][:, 0].cpu().numpy() > 0.5)+1
            assert len(ends) == 1000 and ends[-1] == len(self.source)
            next_angles = raw["next_privileged"][:, 0].cpu().numpy()
            spans = []
            begin = 0
            for end in ends:
                crossing = np.flatnonzero(next_angles[begin:end] > self._target)
                stop = begin+int(crossing[0])+1 if len(crossing) else int(end)
                spans.append(np.arange(begin, stop))
                begin = int(end)
            indices = torch.as_tensor(np.concatenate(spans), device=self.source.device)
            raw = {key: value[indices] for key, value in raw.items()}
        prepared = dict(raw)
        if self.variant in ("B", "D", "E"):
            prepared["reward"] = relabel(raw["reward"], raw["privileged"][:, :1],
                                          raw["next_privileged"][:, :1], raw["next_privileged"][:, 1:2], self.dt)
        if self.variant == "E":
            full = raw["next_privileged"][:, :1] > 1.0
            stage = raw["next_privileged"][:, :1] > self._target
            prepared["reward"] += 600.0*self.dt*(stage.float()-full.float())
            prepared["done"] = torch.maximum(raw["done"], stage.float())
        prepared["robot"] = observation(raw["robot"], raw["privileged"], self.variant, self._target)
        prepared["next_robot"] = observation(raw["next_robot"], raw["next_privileged"], self.variant, self._target)
        self.prepared = prepared

    def __len__(self):
        return self.prepared["action"].shape[0]

    def sample(self, count):
        indices = torch.randint(len(self), (count,), device=self.source.device)
        return {key: value[indices] for key, value in self.prepared.items()}
