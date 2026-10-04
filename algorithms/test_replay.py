import torch

from replay import ReplayBuffer, mixed_sample


def test_wrap_and_mix():
    online = ReplayBuffer(8, 3, 2, 4, device="cpu")
    offline = ReplayBuffer(8, 3, 2, 4, device="cpu")
    batch = {
        "robot": torch.ones(12, 3),
        "privileged": torch.ones(12, 2),
        "action": torch.ones(12, 4),
        "reward": torch.ones(12, 1),
        "next_robot": torch.ones(12, 3),
        "next_privileged": torch.ones(12, 2),
        "done": torch.zeros(12, 1),
    }
    online.add(batch)
    offline.add({key: -value for key, value in batch.items()})
    assert len(online) == len(offline) == 8
    sample = mixed_sample(online, offline, 10, 0.3)
    assert sample["action"].shape == (10, 4)
    assert (sample["reward"] > 0).sum() == 7
    assert (sample["reward"] < 0).sum() == 3


if __name__ == "__main__":
    test_wrap_and_mix()
    print("Replay wraparound and offline mixture: PASS")
