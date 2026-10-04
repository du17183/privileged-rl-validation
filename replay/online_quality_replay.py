"""Online replay with frozen early-trajectory quality scoring and a rank gate."""

from pathlib import Path

import h5py
import numpy as np
import torch

from algorithms.replay import FIELDS, ReplayBuffer
from replay.quality_weighted_replay import normalized_weights
from trajectory_quality.quality_model import TrajectoryQualityModel
from value_analysis.metrics import safe_corr, success_ranking_accuracy


class OnlineQualityReplay(ReplayBuffer):
    def __init__(self, capacity, robot_dim, privileged_dim, action_dim, num_envs,
                 quality_checkpoint, device="cuda", score_horizon=100,
                 collection_window=10000, gate_mode="success"):
        super().__init__(capacity, robot_dim, privileged_dim, action_dim, device=device)
        self.num_envs = num_envs
        self.score_horizon = score_horizon
        self.collection_window = collection_window
        if gate_mode not in ("success", "return", "offline"):
            raise ValueError("gate_mode must be success, return or offline")
        self.gate_mode = gate_mode
        self.pending = [[] for _ in range(num_envs)]
        self.episode_id = torch.full((capacity,), -1, dtype=torch.int64, device=self.device)
        self.quality_score = torch.full((capacity,), float("nan"), device=self.device)
        self.weights = torch.ones(capacity, device=self.device)
        self.episode_summaries = []
        self.active = False
        self.online_rank_auc = float("nan")
        self.online_return_rho = float("nan")
        self.completed = 0
        if quality_checkpoint is None:
            self.model = None
        else:
            saved = torch.load(quality_checkpoint, map_location="cpu", weights_only=False)
            if saved.get("prefix_steps") != score_horizon:
                raise ValueError("Quality model horizon does not match online scoring")
            self.model = TrajectoryQualityModel().to(self.device)
            self.model.load_state_dict(saved["model"])
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.center = torch.as_tensor(saved["input_center"], device=self.device)
            self.scale = torch.as_tensor(saved["input_scale"], device=self.device)

    @torch.no_grad()
    def add_step(self, transition, done, successes, env_steps):
        if len(transition["action"]) != self.num_envs:
            raise ValueError("A vector step must include every environment")
        old_cursor = self.cursor
        self.add(transition)
        indices = (torch.arange(self.num_envs, device=self.device) + old_cursor) % self.capacity
        self.weights[indices] = 1.0
        for env_id in range(self.num_envs):
            index = int(indices[env_id])
            self.pending[env_id].append(index)
            if bool(done[env_id]):
                self._finish(env_id, bool(successes[env_id]), env_steps)

    @torch.no_grad()
    def _finish(self, env_id, success, env_steps):
        index = torch.tensor(self.pending[env_id], device=self.device, dtype=torch.long)
        self.pending[env_id] = []
        if len(index) == 0:
            return
        reward = self.data["reward"][index, 0]
        running = 0.0
        for value in reward.flip(0):
            running = float(value) + 0.99 * running
        if self.model is None:
            score = float("nan")
        else:
            sample = torch.linspace(0, self.score_horizon - 1, 32, device=self.device).long()
            sample = sample.clamp(max=len(index) - 1)
            sequence = torch.cat((self.data["robot"][index[sample]],
                                  self.data["action"][index[sample]]), dim=1)
            normalized = (sequence - self.center) / self.scale
            logit, _ = self.model(normalized.unsqueeze(0))
            score = float(torch.sigmoid(logit[0, 0]))
        self.episode_id[index] = self.completed
        self.quality_score[index] = score
        self.episode_summaries.append({"episode_id": self.completed, "score": score,
                                       "success": int(success), "return": running,
                                       "length": len(index),
                                       "source": env_steps // self.collection_window})
        self.completed += 1

    @torch.no_grad()
    def refresh_weights(self):
        self.active = False
        self.online_rank_auc = float("nan")
        self.online_return_rho = float("nan")
        self.weights[:self.size].fill_(1.0)
        minimum = 32 if self.gate_mode == "offline" else 64
        if self.model is None or len(self.episode_summaries) < minimum:
            return
        recent = self.episode_summaries[-128:]
        scores = np.asarray([row["score"] for row in recent], dtype=float)
        success = np.asarray([row["success"] for row in recent], dtype=bool)
        returns = np.asarray([row["return"] for row in recent], dtype=float)
        self.online_return_rho = safe_corr(scores, returns, "spearman")
        if success.sum() >= 10 and (~success).sum() >= 10:
            self.online_rank_auc = success_ranking_accuracy(scores, success)
        if self.gate_mode != "offline":
            gate_value = self.online_rank_auc if self.gate_mode == "success" else self.online_return_rho
            threshold = 0.65 if self.gate_mode == "success" else 0.20
            if not np.isfinite(gate_value) or gate_value < threshold:
                return
        all_scores = np.asarray([row["score"] for row in self.episode_summaries], dtype=float)
        all_sources = np.asarray([row["source"] for row in self.episode_summaries])
        episode_weights = normalized_weights(all_scores, all_sources)
        indices = self.episode_id[:self.size]
        complete = indices >= 0
        lookup = torch.as_tensor(episode_weights, dtype=torch.float32, device=self.device)
        self.weights[:self.size][complete] = lookup[indices[complete]]
        self.active = True

    def sample(self, count, mode="uniform"):
        if mode == "uniform" or not self.active:
            return super().sample(count)
        if mode != "weighted":
            raise ValueError(f"Unknown online replay mode: {mode}")
        index = torch.multinomial(self.weights[:self.size], count, replacement=True)
        return {key: value[index] for key, value in self.data.items()}

    @torch.no_grad()
    def export_hdf5(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            for key in FIELDS:
                h5.create_dataset(key, data=self.data[key][:self.size].cpu().numpy(), compression="gzip")
            for key in ("episode_id", "quality_score", "weights"):
                h5.create_dataset(key, data=getattr(self, key)[:self.size].cpu().numpy(), compression="gzip")
            h5.attrs["completed_episodes"] = self.completed
            h5.attrs["score_horizon"] = self.score_horizon
            h5.attrs["online_rank_gate"] = 0.65
            h5.attrs["gate_mode"] = self.gate_mode
            h5.attrs["online_return_gate"] = 0.20
