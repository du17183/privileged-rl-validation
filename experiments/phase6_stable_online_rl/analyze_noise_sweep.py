"""Five-seed paired audit of B3 execution-noise sensitivity, no training."""

import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase6_stable_online_rl"
T95 = 2.7764451051977987
MODES = ("deterministic", "raw_stochastic", "cap_0p135", "cap_0p05", "cap_0p01")


def read_one(seed, mode):
    suffix = {"deterministic": "", "raw_stochastic": "_stochastic",
              "cap_0p135": "_stochastic_cap0p135",
              "cap_0p05": "_stochastic_cap0p05",
              "cap_0p01": "_stochastic_cap0p01"}[mode]
    path = OUT / f"heldout_P6B3_seed{seed}_best{suffix}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def summarize(values, bounded=False):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = T95 * float(values.std(ddof=1)) / np.sqrt(5)
    low, high = mean-radius, mean+radius
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return mean, low, high


def pvalue(values):
    values = np.asarray(values, dtype=float)
    threshold = abs(values.mean())-1e-12
    return float(np.mean([abs(np.mean(values*np.asarray(signs))) >= threshold
                          for signs in itertools.product((-1, 1), repeat=5)]))


def write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    observations = {(mode, seed): read_one(seed, mode)
                    for mode in MODES for seed in range(5)}
    per_seed = [(mode, seed,
                 float(observations[(mode, seed)]["heldout_success_rate"]),
                 float(observations[(mode, seed)]["reset_mean_action_std"]),
                 int(observations[(mode, seed)]["eval_env_steps"]))
                for mode in MODES for seed in range(5)]
    write(OUT / "noise_sweep_per_seed.csv",
          ("mode", "seed", "success_rate", "uncapped_reset_action_std",
           "eval_env_steps"), per_seed)
    aggregate = []
    for mode in MODES:
        values = [float(observations[(mode, seed)]["heldout_success_rate"])
                  for seed in range(5)]
        aggregate.append((mode, *summarize(values, bounded=True)))
    write(OUT / "noise_sweep_aggregate.csv",
          ("mode", "mean_success", "ci95_low", "ci95_high"), aggregate)
    paired = []
    for mode in MODES[2:]:
        differences = [float(observations[(mode, seed)]["heldout_success_rate"]) -
                       float(observations[("raw_stochastic", seed)]["heldout_success_rate"])
                       for seed in range(5)]
        mean, low, high = summarize(differences)
        paired.append((mode, "raw_stochastic", mean, max(-1.0, low), min(1.0, high),
                       pvalue(differences)))
    write(OUT / "noise_sweep_paired.csv",
          ("treatment", "control", "mean_difference", "ci95_low", "ci95_high",
           "exact_two_sided_signflip_p"), paired)
    (OUT / "noise_sweep_manifest.json").write_text(json.dumps({
        "training_updates": 0, "checkpoint": "B3 best from fixed evaluation",
        "episodes_per_seed_mode": 64, "total_eval_env_steps": sum(r[4] for r in per_seed),
        "interpretation": "Execution-only noise cap; not online RL improvement."
    }, indent=2))
    print("Noise sweep complete")
    for row in aggregate:
        print(row)
    for row in paired:
        print(row)


if __name__ == "__main__":
    main()
