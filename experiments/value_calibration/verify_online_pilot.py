"""Verify paired initialization, run completeness, and actual replay intervention."""

import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"
CKPT = ROOT / "checkpoints" / "value_calibration" / "online_pilot"
ARMS = ("uniform", "quality", "quality_return", "quality_offline")


def state_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)


def main():
    rows = []
    for seed in range(5):
        reference = None
        first_success = None
        for arm in ARMS:
            run = f"{arm}_seed{seed}"
            with (OUT / f"eval_{run}.csv").open(newline="", encoding="utf-8") as stream:
                evaluation = list(csv.DictReader(stream))
            steps = [int(row["env_steps"]) for row in evaluation]
            if len(steps) != 11 or steps[0] != 0 or steps[-1] != 100000:
                raise RuntimeError(f"Incomplete evaluation curve: {run}")
            state = torch.load(CKPT / run / "step_0.pt", map_location="cpu", weights_only=False)
            if reference is None:
                reference = state
                first_success = float(evaluation[0]["success_rate"])
            elif not state_equal(reference["actor"], state["actor"]) or \
                    not state_equal(reference["critic"], state["critic"]) or \
                    float(evaluation[0]["success_rate"]) != first_success:
                raise RuntimeError(f"Paired initial state differs: {run}")
            activated = [int(row["env_steps"]) for row in evaluation
                         if int(row["weighted_active"])]
            rows.append({"seed": seed, "arm": arm, "initial_actor_critic_exact": True,
                         "evaluation_rows": len(steps),
                         "first_weighted_step": min(activated) if activated else -1,
                         "weighted_eval_count": len(activated),
                         "initial_success": first_success})
    if not all(row["weighted_eval_count"] >= 1 for row in rows
               if row["arm"] == "quality_offline"):
        raise RuntimeError("Offline-gated intervention never activated in at least one seed")
    path = OUT / "protocol_verification.json"
    path.write_text(json.dumps({"passed": True, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Verified {len(rows)} runs, exact paired step-0 actor/critic, and all offline-gated interventions")


if __name__ == "__main__":
    main()
