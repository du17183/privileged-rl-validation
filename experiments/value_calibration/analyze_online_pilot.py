"""Paired five-seed summary of the gate-controlled 100k Door replay pilot."""

import csv
import itertools
from pathlib import Path

import numpy as np

from trajectory_quality.ranking_eval import interval, write_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"


def load(arm, seed):
    path = OUT / f"eval_{arm}_seed{seed}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steps = np.asarray([int(row["env_steps"]) for row in rows])
    if len(rows) != 11 or steps[0] != 0 or steps[-1] != 100000 or not np.all(np.diff(steps) > 0):
        raise RuntimeError(f"Incomplete 100k pilot run: {path}")
    return rows, steps


def exact_signflip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    possible = [abs(np.mean(values * signs))
                for signs in itertools.product((-1, 1), repeat=len(values))]
    return float(np.mean(np.asarray(possible) >= observed - 1e-12))


def main():
    metrics, curves = [], []
    for arm in ("uniform", "quality", "quality_return", "quality_offline"):
        for seed in range(5):
            rows, steps = load(arm, seed)
            success = np.asarray([float(row["success_rate"]) for row in rows])
            rewards = np.asarray([float(row["mean_return"]) for row in rows])
            best_index = int(success.argmax())
            metrics.append({"arm": arm, "seed": seed,
                            "auc": float(np.trapz(success, steps) / steps[-1]),
                            "reward_auc": float(np.trapz(rewards, steps) / steps[-1]),
                            "final_success": float(success[-1]),
                            "best_success": float(success[best_index]),
                            "best_step": int(steps[best_index]),
                            "degradation": float(success[best_index] - success[-1]),
                            "weighted_eval_fraction": float(np.mean([
                                int(row["weighted_active"]) for row in rows[1:]])),
                            "completed_online_episodes": int(rows[-1]["complete_online_episodes"]),
                            "total_eval_interactions": int(rows[-1]["eval_env_steps"]),
                            "wall_time_s": float(rows[-1]["wall_time_s"])})
            curves.extend({"arm": arm, "seed": seed, "env_steps": int(step),
                           "success_rate": float(row["success_rate"]),
                           "mean_return": float(row["mean_return"]),
                           "online_rank_auc": float(row["online_rank_auc"]),
                           "weighted_active": int(row["weighted_active"])}
                          for step, row in zip(steps, rows))
    write_csv(OUT / "pilot_per_seed.csv", metrics)
    write_csv(OUT / "pilot_curve.csv", curves)
    aggregate = []
    for metric in ("auc", "final_success", "best_success", "degradation",
                   "reward_auc", "weighted_eval_fraction"):
        uniform = np.asarray([row[metric] for row in metrics if row["arm"] == "uniform"])
        quality = np.asarray([row[metric] for row in metrics if row["arm"] == "quality"])
        quality_return = np.asarray([row[metric] for row in metrics if row["arm"] == "quality_return"])
        quality_offline = np.asarray([row[metric] for row in metrics if row["arm"] == "quality_offline"])
        for arm, values in (("uniform", uniform), ("quality", quality),
                            ("quality_return", quality_return),
                            ("quality_offline", quality_offline),
                            ("quality_minus_uniform", quality - uniform),
                            ("quality_return_minus_uniform", quality_return - uniform),
                            ("quality_offline_minus_uniform", quality_offline - uniform)):
            mean, lo, hi = interval(values)
            if metric != "reward_auc" and not arm.endswith("_minus_uniform"):
                lo, hi = max(0.0, lo), min(1.0, hi)
            aggregate.append({"metric": metric, "arm_or_paired_difference": arm,
                              "mean": mean, "ci95_low": lo, "ci95_high": hi,
                              "paired_signflip_p": exact_signflip(values) if arm.endswith("_minus_uniform") else float("nan")})
    write_csv(OUT / "pilot_aggregate.csv", aggregate)
    for row in aggregate:
        if row["metric"] in ("auc", "final_success", "weighted_eval_fraction"):
            print(row)


if __name__ == "__main__":
    main()
