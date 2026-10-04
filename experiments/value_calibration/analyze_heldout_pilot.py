"""Five-seed independent actor verification and paired 95% intervals."""

import csv
from pathlib import Path

import numpy as np

from experiments.value_calibration.analyze_online_pilot import exact_signflip
from trajectory_quality.ranking_eval import interval, write_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"
ARMS = ("uniform", "quality", "quality_return", "quality_offline")


def main():
    rows = []
    for arm in ARMS:
        for seed in range(5):
            with (OUT / f"heldout_{arm}_seed{seed}.csv").open(newline="", encoding="utf-8") as stream:
                checks = {row["mode"]: row for row in csv.DictReader(stream)}
            if set(checks) != {"best", "final"}:
                raise RuntimeError(f"Incomplete independent verification: {arm} seed{seed}")
            best = float(checks["best"]["heldout_success_64"])
            final = float(checks["final"]["heldout_success_64"])
            rows.append({"arm": arm, "seed": seed, "best_heldout_success": best,
                         "final_heldout_success": final,
                         "heldout_degradation": best - final,
                         "best_checkpoint_steps": int(checks["best"]["checkpoint_steps"]),
                         "total_heldout_interactions": sum(int(row["heldout_interactions"])
                                                           for row in checks.values())})
    write_csv(OUT / "heldout_per_seed.csv", rows)
    aggregate = []
    for metric in ("best_heldout_success", "final_heldout_success", "heldout_degradation"):
        baseline = np.asarray([row[metric] for row in rows if row["arm"] == "uniform"])
        for arm in ARMS:
            values = np.asarray([row[metric] for row in rows if row["arm"] == arm])
            mean, lo, hi = interval(values)
            lo, hi = max(0.0, lo), min(1.0, hi)
            aggregate.append({"metric": metric, "arm_or_paired_difference": arm,
                              "mean": mean, "ci95_low": lo, "ci95_high": hi,
                              "paired_signflip_p": float("nan")})
            if arm != "uniform":
                diff = values - baseline
                mean, lo, hi = interval(diff)
                aggregate.append({"metric": metric,
                                  "arm_or_paired_difference": f"{arm}_minus_uniform",
                                  "mean": mean, "ci95_low": lo, "ci95_high": hi,
                                  "paired_signflip_p": exact_signflip(diff)})
    write_csv(OUT / "heldout_aggregate.csv", aggregate)
    for row in aggregate:
        print(row)


if __name__ == "__main__":
    main()
