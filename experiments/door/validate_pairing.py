"""Verify paired A/B/D actor initialization and step-zero evaluation equality."""
import csv
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
records = []
for seed in range(5):
    states = {}
    evals = {}
    for variant in ("A", "B", "D"):
        path = ROOT / "checkpoints/door" / f"{variant}_seed{seed}" / "step_0.pt"
        states[variant] = torch.load(path, map_location="cpu", weights_only=False)["actor"]
        with (ROOT / "results/door" / f"eval_{variant}_seed{seed}.csv").open(newline="", encoding="utf-8") as stream:
            evals[variant] = next(csv.DictReader(stream))["success_rate"]
    difference = max(float((states["A"][key] - states[variant][key]).abs().max())
                     for variant in ("B", "D") for key in states["A"])
    if difference != 0 or len(set(evals.values())) != 1:
        raise RuntimeError(f"Paired seed mismatch: {seed} {difference} {evals}")
    records.append({"seed": seed, "actor_max_parameter_difference": difference,
                    "step_zero_success_rate": float(evals["A"])})
out = ROOT / "results/door/pairing_validation.json"
out.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
