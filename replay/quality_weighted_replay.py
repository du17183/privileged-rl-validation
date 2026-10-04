"""Trajectory-level replay interface for later online Door experiments.

The immutable robot/action trajectory is scored after completion. Privileged
state and fixture outcome are metadata; neither is passed to the policy.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata


@dataclass
class Trajectory:
    key: str
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    return_to_go: np.ndarray
    value: float
    success: bool
    quality_score: float

    def as_dict(self):
        return {"state": self.state, "action": self.action,
                "reward": self.reward, "next_state": self.next_state,
                "return": self.return_to_go, "value": self.value,
                "success": self.success, "quality_score": self.quality_score}


def load_trajectory(path: Path, group_name: str, value=float("nan"),
                    quality_score=float("nan")) -> Trajectory:
    with h5py.File(path, "r") as h5:
        group = h5[group_name]
        return Trajectory(
            key=f"{path.stem}/{group_name}",
            state=group["observation"][:],
            action=group["action"][:],
            reward=group["reward"][:],
            next_state=group["next_observation"][:],
            return_to_go=group["mc_return_to_go"][:],
            value=float(value),
            success=bool(group.attrs["success"]),
            quality_score=float(quality_score),
        )


def normalized_weights(scores, sources=None, lower=0.5, upper=2.0):
    """Bounded percentile weights with equal mass per trajectory source.

    A source can be a rollout policy or a recent online collection window.
    Ranking within each source limits policy-mixture confounding. Weights are
    normalized to mean one; their exact expectation is used in offline tests.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Replay scores must be finite and nonempty")
    if not (0 < lower <= 1 <= upper):
        raise ValueError("Require 0 < lower <= 1 <= upper")
    sources = np.zeros(len(scores), dtype=np.int64) if sources is None else np.asarray(sources)
    if len(sources) != len(scores):
        raise ValueError("Source labels and scores have different lengths")
    weights = np.ones(len(scores), dtype=np.float64)
    for source in np.unique(sources):
        index = np.flatnonzero(sources == source)
        if len(index) <= 1 or np.ptp(scores[index]) < 1e-12:
            continue
        rank = rankdata(scores[index], method="average") - 1
        percentile = rank / (len(index) - 1)
        weights[index] = lower + (upper - lower) * percentile
        weights[index] /= weights[index].mean()
    weights /= weights.mean()
    return weights


class TrajectoryReplay:
    def __init__(self, keys, sources, values, quality_scores):
        self.keys = np.asarray(keys)
        self.sources = np.asarray(sources)
        self.values = np.asarray(values, dtype=float)
        self.quality_scores = np.asarray(quality_scores, dtype=float)
        if not (len(self.keys) == len(self.sources) == len(self.values) == len(self.quality_scores)):
            raise ValueError("Trajectory metadata lengths disagree")

    def probabilities(self, mode="uniform"):
        if mode == "uniform":
            weights = np.ones(len(self.keys), dtype=float)
        elif mode == "value":
            weights = normalized_weights(self.values, self.sources)
        elif mode == "quality":
            weights = normalized_weights(self.quality_scores, self.sources)
        else:
            raise ValueError(f"Unknown replay mode: {mode}")
        return weights / weights.sum()

    def sample_indices(self, batch_size, mode="uniform", rng=None):
        generator = rng or np.random.default_rng()
        return generator.choice(len(self.keys), size=batch_size, replace=True,
                                p=self.probabilities(mode))
