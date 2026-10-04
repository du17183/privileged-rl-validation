"""Check critic-value weighting, trajectory normalization and uniform support."""
import json
from pathlib import Path
import torch
from replay.value_weighted_replay import ValueWeightedReplay


class ToyCritic:
    def __call__(self, obs, action):
        q = obs[:, -1:]
        return q, q + 1


buffer = ValueWeightedReplay(100, 2, 1, 1, device="cpu")
for episode in range(18):
    score = 10.0 if episode == 16 else 0.0
    batch = {
        "robot": torch.zeros(2, 2), "privileged": torch.full((2, 1), score),
        "action": torch.zeros(2, 1), "reward": torch.zeros(2, 1),
        "next_robot": torch.zeros(2, 2), "next_privileged": torch.full((2, 1), score),
        "done": torch.tensor([[0.], [1.]]),
    }
    indices = buffer.add(batch, [episode, episode])
    buffer.finalize(episode, indices, ToyCritic(), succeeded=(episode == 17))
buffer.sampling_stats()
p = buffer._distribution
high = float(p[32:34].sum())
low = float(p[34:36].sum())
assert high > 3 * low, (high, low)
assert float(p.min()) >= 0.2 / 36 - 1e-8
assert buffer.sample_experience(8)["value"].shape == (8, 1)
result = {"high_value_failure_probability": high,
          "low_value_success_probability": low,
          "success_label_used_for_weight": False,
          "uniform_support_floor": 0.2}
path = Path(__file__).resolve().parents[1] / "results/door/replay_unit_validation.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(result)
