"""GPU replay buffer with explicit privileged and next-state fields."""

import torch


FIELDS = (
    "robot", "privileged", "action", "reward", "next_robot", "next_privileged", "done"
)


class ReplayBuffer:
    def __init__(self, capacity, robot_dim, privileged_dim, action_dim, device="cuda"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        dims = {
            "robot": robot_dim,
            "privileged": privileged_dim,
            "action": action_dim,
            "reward": 1,
            "next_robot": robot_dim,
            "next_privileged": privileged_dim,
            "done": 1,
        }
        self.data = {
            key: torch.empty((self.capacity, dim), dtype=torch.float32, device=self.device)
            for key, dim in dims.items()
        }
        self.cursor = 0
        self.size = 0

    def __len__(self):
        return self.size

    @torch.no_grad()
    def add(self, batch):
        if set(batch) != set(FIELDS):
            raise KeyError(f"Expected {FIELDS}; got {tuple(batch)}")
        count = batch["action"].shape[0]
        if count == 0:
            return
        if count > self.capacity:
            batch = {key: value[-self.capacity:] for key, value in batch.items()}
            count = self.capacity
        first = min(count, self.capacity - self.cursor)
        second = count - first
        for key in FIELDS:
            value = torch.as_tensor(batch[key], dtype=torch.float32, device=self.device)
            if value.shape != (count, self.data[key].shape[1]):
                raise ValueError(f"Bad shape for {key}: {tuple(value.shape)}")
            self.data[key][self.cursor:self.cursor + first].copy_(value[:first])
            if second:
                self.data[key][:second].copy_(value[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count):
        if self.size == 0:
            raise ValueError("Replay buffer is empty")
        idx = torch.randint(self.size, (count,), device=self.device)
        return {key: value[idx] for key, value in self.data.items()}


def mixed_sample(online, offline, batch_size, offline_fraction):
    offline_count = round(batch_size * offline_fraction) if len(offline) else 0
    online_count = batch_size - offline_count
    if not len(online):
        offline_count, online_count = batch_size, 0
    parts = []
    if online_count:
        parts.append(online.sample(online_count))
    if offline_count:
        parts.append(offline.sample(offline_count))
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in FIELDS}
