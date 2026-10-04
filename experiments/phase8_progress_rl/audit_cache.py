"""Prove cached expert preprocessing preserves every primary minibatch field."""
import json
import math
from pathlib import Path
import torch
import argparse
from offline_rl.baseline import load_demonstrations
from progress_rl.agent import ExpertView
from progress_rl.progress_state import observation
from progress_rl.progress_reward import relabel
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(991)
    expected_noise = torch.randn((32, 7), device=args.device)
    torch.manual_seed(991)
    diagnostic_rng = torch.Generator(device=args.device).manual_seed(1991)
    torch.randn((8, 32, 7), device=args.device, generator=diagnostic_rng)
    actual_noise = torch.randn((32, 7), device=args.device)
    assert torch.equal(expected_noise, actual_noise), "Density diagnostics changed execution RNG"
    source = load_demonstrations(ROOT/"door_dataset"/"door_expert_1000.h5", args.device)
    counts = {}
    for arm in ("A", "B", "C", "D"):
        view = ExpertView(source, arm, 1/60)
        for seed in range(5):
            torch.manual_seed(seed)
            original = source.sample(256)
            if arm in ("B", "D"):
                original["reward"] = relabel(original["reward"], original["privileged"][:, :1],
                                              original["next_privileged"][:, :1], original["next_privileged"][:, 1:2], 1/60)
            original["robot"] = observation(original["robot"], original["privileged"], arm)
            original["next_robot"] = observation(original["next_robot"], original["next_privileged"], arm)
            torch.manual_seed(seed)
            cached = view.sample(256)
            assert all(torch.equal(original[key], cached[key]) for key in original), (arm, seed)
        counts[arm] = len(view)
    for degrees in (10, 30, 50):
        view = ExpertView(source, "E", 1/60, math.radians(degrees))
        sample = view.sample(4096)
        assert (sample["privileged"][:, 0] <= math.radians(degrees)).all()
        assert torch.equal(sample["done"].bool(), sample["next_privileged"][:, :1] > math.radians(degrees))
        counts[f"E_{degrees}"] = len(view)
    result = dict(primary_minibatches_bitwise_equal=True, execution_rng_unchanged=True,
                  device=args.device, transition_counts=counts,
                  applies_to="Future starts only; ongoing primary learners retain archived original code")
    (ROOT/"results"/"phase8_progress_rl"/"expert_cache_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
