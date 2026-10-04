"""Audit an independent 64-episode final-checkpoint replay for all 24 runs."""

import csv
import itertools
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for variant in "ABC":
        for seed in range(8):
            path = root / "results" / f"heldout_{variant}_seed{seed}.json"
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["variant"] != variant or item["episodes"] != 64 or item["env_steps"] != 200000:
                raise ValueError(f"Incomplete or mismatched held-out replay: {path}")
            rows.append({
                "variant": variant, "seed": seed, "episodes": item["episodes"],
                "success_rate": item["success_rate"],
                "grasp_rate": item["grasp_rate"],
                "contact_assisted_success_rate": item["contact_assisted_success_rate"],
                "both_fingers_contact_steps_mean": item["both_fingers_contact_steps_mean"],
                "simulator_interactions": item["simulator_interactions"],
            })
    with (root / "results/heldout_final.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evaluation_seed": 901,
        "episodes_per_final_checkpoint": 64,
        "additional_evaluation_interactions": sum(row["simulator_interactions"] for row in rows),
        "variants": {},
    }
    for variant in "ABC":
        group = [row for row in rows if row["variant"] == variant]
        rates = [row["success_rate"] for row in group]
        contacts = [row["contact_assisted_success_rate"] for row in group]
        summary["variants"][variant] = {
            "success_mean": statistics.mean(rates),
            "success_sd": statistics.stdev(rates),
            "contact_assisted_success_mean": statistics.mean(contacts),
            "per_seed_success": {str(row["seed"]): row["success_rate"] for row in group},
        }
    a = [row["success_rate"] for row in rows if row["variant"] == "A"]
    b = [row["success_rate"] for row in rows if row["variant"] == "B"]
    differences = [b_value - a_value for a_value, b_value in zip(a, b)]
    observed = abs(statistics.mean(differences))
    flips = [
        abs(statistics.mean(d * sign for d, sign in zip(differences, signs)))
        for signs in itertools.product((-1, 1), repeat=8)
    ]
    summary["paired_B_minus_A"] = {
        "mean_final_success_gain": statistics.mean(differences),
        "exact_two_sided_sign_flip_p": sum(value >= observed - 1e-12 for value in flips) / len(flips),
    }
    (root / "results/heldout_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
