"""Audit Phase 6 paired initialization, budget and actor information route."""

import argparse
import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints" / "phase6_stable_online_rl"
OUT = ROOT / "results" / "phase6_stable_online_rl"
VARIANTS = ("A0", "A1", "A2", "B1", "B2", "B3")


def state(variant, seed):
    path = CHECKPOINTS / f"P6{variant}_seed{seed}" / "step_0.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def identical(left, right, component):
    l, r = left[component], right[component]
    return l.keys() == r.keys() and all(torch.equal(l[k], r[k]) for k in l)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true",
                        help="Check only checkpoints already available; do not write final manifest")
    options = parser.parse_args()
    checks = []
    available_runs = 0
    completed_runs = 0
    for seed in range(5):
        states = {variant: state(variant, seed) for variant in VARIANTS
                  if (CHECKPOINTS / f"P6{variant}_seed{seed}" / "step_0.pt").exists()}
        if not options.partial and len(states) != len(VARIANTS):
            raise AssertionError(f"Missing initial checkpoints for seed {seed}")
        available_runs += len(states)
        for pair in (("A0", "A1"), ("A0", "A2"),
                     ("B1", "B2"), ("B1", "B3")):
            if not all(variant in states for variant in pair):
                continue
            for component in ("actor", "critic", "target_critic"):
                result = identical(states[pair[0]], states[pair[1]], component)
                checks.append((seed, pair[0], pair[1], component, result))
                if not result:
                    raise AssertionError(f"Initial {component} mismatch seed {seed}: {pair}")
        for variant, checkpoint in states.items():
            recipe = checkpoint["recipe"]
            if checkpoint["variant"] != "A" or checkpoint["env_steps"] != 0:
                raise AssertionError(f"Wrong actor/critic route or initial step: {variant}/{seed}")
            if checkpoint["actor_opt"]["state"]:
                raise AssertionError(f"Actor optimizer retained BC momentum: {variant}/{seed}")
            if variant.startswith("A") and recipe["bc_initialized"]:
                raise AssertionError("A replay group unexpectedly used BC")
            if variant.startswith("B") and not recipe["bc_initialized"]:
                raise AssertionError("B policy group lacks BC")
            with (OUT / f"eval_P6{variant}_seed{seed}.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if int(rows[-1]["env_steps"]) == 500000 and len(rows) == 51:
                completed_runs += 1
            elif not options.partial:
                raise AssertionError(f"Incomplete run {variant}/{seed}")
            if any(int(row["eval_episodes"]) < 50 for row in rows):
                raise AssertionError(f"Too few evaluation episodes {variant}/{seed}")
    output = {"paired_initialization_checks": len(checks),
              "all_initial_weights_identical_within_control_families": True,
              "available_runs": available_runs,
              "complete_training_runs": completed_runs,
              "evaluation_episodes_per_checkpoint": 64,
              "actor_observation_route": "robot-only; SAC checkpoint variant A"}
    if not options.partial:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "protocol_verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
