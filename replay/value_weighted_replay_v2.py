"""Trajectory-aware online replay with calibrated, bounded value sampling.

The robot-only Q estimate is used to rank completed trajectories only when
its rank correlation with observed returns on recent episodes is positive.
This prevents an uncalibrated critic from silently dominating replay.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import torch

from algorithms.replay import FIELDS, ReplayBuffer


@dataclass
class TrajectoryObject:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    return_to_go: torch.Tensor
    estimated_value: torch.Tensor
    success: float
    quality_score: float

    def as_dict(self):
        return {"state": self.state, "action": self.action, "reward": self.reward,
                "next_state": self.next_state, "return": self.return_to_go,
                "value": self.estimated_value, "success": self.success,
                "quality_score": self.quality_score}


class ValueWeightedTrajectoryReplay(ReplayBuffer):
    """Ring buffer with complete-episode labels and uniform/weighted modes."""

    def __init__(self, capacity, robot_dim, privileged_dim, action_dim,
                 num_envs, device="cuda", gamma=0.99, beta=1.0):
        super().__init__(capacity, robot_dim, privileged_dim, action_dim, device=device)
        self.num_envs = num_envs
        self.gamma = gamma
        self.beta = beta
        self.pending = [[] for _ in range(num_envs)]
        self.episode_id = torch.full((capacity,), -1, dtype=torch.int64, device=self.device)
        self.return_to_go = torch.full((capacity,), float("nan"), device=self.device)
        self.estimated_value = torch.full((capacity,), float("nan"), device=self.device)
        self.trajectory_value = torch.full((capacity,), float("nan"), device=self.device)
        self.success = torch.full((capacity,), float("nan"), device=self.device)
        self.quality_score = torch.full((capacity,), float("nan"), device=self.device)
        self.weights = torch.ones(capacity, device=self.device)
        self.episode_summaries = []
        self.complete_count = 0
        self.calibration_rho = float("nan")
        self.weighting_active = False

    @torch.no_grad()
    def add_step(self, transition, done, successes, agent):
        if transition["action"].shape[0] != self.num_envs:
            raise ValueError("One vector step must contain every environment")
        old_cursor = self.cursor
        self.add(transition)
        indices = (torch.arange(self.num_envs, device=self.device) + old_cursor) % self.capacity
        for env_id in range(self.num_envs):
            idx = int(indices[env_id])
            self.pending[env_id].append(idx)
            if bool(done[env_id]):
                self._finish(env_id, float(successes[env_id]), agent)

    def discard_pending(self):
        """Validation resets interrupt episodes; never stitch across them."""
        self.pending = [[] for _ in range(self.num_envs)]

    @torch.no_grad()
    def _finish(self, env_id, success, agent):
        indices = torch.tensor(self.pending[env_id], dtype=torch.long, device=self.device)
        self.pending[env_id] = []
        if not len(indices):
            return
        reward = self.data["reward"][indices, 0]
        returns = torch.empty_like(reward)
        running = torch.zeros((), device=self.device)
        for t in range(len(indices) - 1, -1, -1):
            running = reward[t] + self.gamma * running
            returns[t] = running
        robot, action = self.data["robot"][indices], self.data["action"][indices]
        values = []
        for start in range(0, len(indices), 1024):
            stop = start + 1024
            z = agent.critic_input(robot[start:stop], None)
            q1, q2 = agent.critic(z, action[start:stop])
            values.append(torch.minimum(q1, q2).flatten())
        q = torch.cat(values)
        # Compare like with like: Q at the trajectory start against its
        # discounted return from the same transition.
        ep_value = float(q[0])
        ep_return = float(returns[0])
        self.episode_id[indices] = self.complete_count
        self.return_to_go[indices] = returns
        self.estimated_value[indices] = q
        self.trajectory_value[indices] = ep_value
        self.success[indices] = success
        self.quality_score[indices] = 0.5  # replaced by normalized score at refresh
        self.episode_summaries.append({"episode_id": self.complete_count,
                                       "return": ep_return, "value": ep_value,
                                       "success": success, "length": len(indices)})
        self.complete_count += 1

    @torch.no_grad()
    def refresh_weights(self):
        """Calibrate value ranking on recent episodes before weighting replay.

        Spearman >=0.2 over at least 64 completed episodes is required.
        Rank-derived weights are normalized and clipped to [0.5, 2.0].
        Incomplete episodes remain available with unit sampling weight.
        """
        recent = self.episode_summaries[-256:]
        self.weighting_active = False
        self.calibration_rho = float("nan")
        self.weights[:self.size].fill_(1.0)
        completed = self.episode_id[:self.size] >= 0
        if not bool(completed.any()):
            return
        values = self.trajectory_value[:self.size][completed]
        center = torch.median(values)
        scale = torch.quantile(values, 0.75) - torch.quantile(values, 0.25)
        scale = scale.clamp_min(1e-4)
        z = ((values - center) / scale).clamp(-2.0, 2.0)
        self.quality_score[:self.size][completed] = torch.sigmoid(z)
        if len(recent) < 64:
            return
        v = torch.tensor([e["value"] for e in recent], device=self.device)
        r = torch.tensor([e["return"] for e in recent], device=self.device)
        vr = v.argsort().argsort().float()
        rr = r.argsort().argsort().float()
        rho = float(torch.corrcoef(torch.stack((vr, rr)))[0, 1])
        self.calibration_rho = rho
        if not torch.isfinite(torch.tensor(rho)) or rho < 0.2:
            return
        weights = torch.exp(self.beta * z).clamp(0.5, 2.0)
        weights = (weights / weights.mean().clamp_min(1e-8)).clamp(0.5, 2.0)
        self.weights[:self.size][completed] = weights
        self.weighting_active = True

    @torch.no_grad()
    def sample(self, count, mode="uniform"):
        if self.size == 0:
            raise ValueError("Replay buffer is empty")
        if mode == "uniform" or not self.weighting_active:
            idx = torch.randint(self.size, (count,), device=self.device)
        elif mode == "weighted":
            idx = torch.multinomial(self.weights[:self.size], count, replacement=True)
        else:
            raise ValueError(f"Unknown replay mode: {mode}")
        return {key: value[idx] for key, value in self.data.items()}

    @torch.no_grad()
    def trajectory(self, episode_id):
        indices = torch.nonzero(self.episode_id[:self.size] == episode_id,
                                as_tuple=False).flatten()
        if not len(indices):
            raise KeyError(f"Unknown or overwritten trajectory: {episode_id}")
        return TrajectoryObject(
            state=torch.cat((self.data["robot"][indices], self.data["privileged"][indices]), dim=-1),
            action=self.data["action"][indices], reward=self.data["reward"][indices],
            next_state=torch.cat((self.data["next_robot"][indices],
                                  self.data["next_privileged"][indices]), dim=-1),
            return_to_go=self.return_to_go[indices],
            estimated_value=self.estimated_value[indices],
            success=float(self.success[indices[0]]),
            quality_score=float(self.quality_score[indices[0]]))

    @torch.no_grad()
    def export_hdf5(self, path):
        """Persist full transition fields and trajectory metadata for audit/LWD."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            for key in FIELDS:
                h5.create_dataset(key, data=self.data[key][:self.size].cpu().numpy(),
                                  compression="gzip", compression_opts=1)
            for key, first, second in (("state", "robot", "privileged"),
                                       ("next_state", "next_robot", "next_privileged")):
                h5.create_dataset(key, data=torch.cat((self.data[first][:self.size],
                                                       self.data[second][:self.size]), dim=-1).cpu().numpy(),
                                  compression="gzip", compression_opts=1)
            for key in ("episode_id", "return_to_go", "estimated_value", "trajectory_value",
                        "success", "quality_score"):
                h5.create_dataset(key, data=getattr(self, key)[:self.size].cpu().numpy(),
                                  compression="gzip", compression_opts=1)
            h5["return"] = h5["return_to_go"]
            h5["value"] = h5["estimated_value"]
            h5.create_dataset("sampling_weight", data=self.weights[:self.size].cpu().numpy(),
                              compression="gzip", compression_opts=1)
            h5.attrs["complete_episodes"] = self.complete_count
            h5.attrs["schema"] = "state,action,reward,next_state,return,value,success,quality_score"
