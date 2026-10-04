"""Summarize independent best and final checkpoint evaluation across seeds."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.door.analyze import paired_interval, write_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/door"
SEEDS = range(5)
VARIANTS = "ABCD"

data = {}
rows = []
for variant in VARIANTS:
    for seed in SEEDS:
        for mode in ("best", "final"):
            path = OUT / f"heldout_{variant}_seed{seed}_{mode}.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            if int(row["episodes"]) != 64:
                raise RuntimeError(f"Incomplete heldout replay: {path}")
            data[(variant, seed, mode)] = row
            rows.append((variant, seed, mode, int(row["selected_steps"]),
                         float(row["selection_success_rate"]), float(row["heldout_success_rate"]),
                         float(row["heldout_mean_return"]), float(row["heldout_contact_rate"]),
                         int(row["eval_env_steps"])))
write_csv(OUT / "heldout_all.csv",
          ("variant", "seed", "mode", "selected_steps", "selection_success_rate",
           "heldout_success_rate", "heldout_mean_return", "heldout_contact_rate", "eval_env_steps"), rows)

summary = {}
aggregate = []
for variant in VARIANTS:
    summary[variant] = {}
    for mode in ("best", "final"):
        success = np.asarray([float(data[(variant, s, mode)]["heldout_success_rate"]) for s in SEEDS])
        reward = np.asarray([float(data[(variant, s, mode)]["heldout_mean_return"]) for s in SEEDS])
        steps = np.asarray([int(data[(variant, s, mode)]["selected_steps"]) for s in SEEDS])
        item = {"mean_success": float(success.mean()), "sd_success": float(success.std(ddof=1)),
                "mean_reward": float(reward.mean()), "median_selected_online_steps": float(np.median(steps)),
                "eval_env_steps": int(sum(int(data[(variant, s, mode)]["eval_env_steps"]) for s in SEEDS))}
        summary[variant][mode] = item
        aggregate.append((variant, mode, *item.values()))
write_csv(OUT / "heldout_aggregate.csv",
          ("variant", "mode", "mean_success", "sd_success", "mean_reward",
           "median_selected_online_steps", "eval_env_steps"), aggregate)

rng = np.random.default_rng(20260930)
paired = []
for mode in ("best", "final"):
    for left, right in (("B", "A"), ("D", "B"), ("C", "A")):
        x = [float(data[(left, s, mode)]["heldout_success_rate"]) for s in SEEDS]
        y = [float(data[(right, s, mode)]["heldout_success_rate"]) for s in SEEDS]
        mean, lo, hi, p = paired_interval(x, y, rng)
        paired.append((mode, f"{left}-{right}", mean, lo, hi, p))
write_csv(OUT / "heldout_paired_effects.csv",
          ("mode", "contrast", "mean_difference", "bootstrap_ci_2p5",
           "bootstrap_ci_97p5", "exact_signflip_p_two_sided"), paired)
summary["paired_effects"] = paired
(OUT / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
