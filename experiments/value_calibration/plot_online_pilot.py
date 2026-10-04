"""Static scientific figures for the five-seed 100k Door replay pilot."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trajectory_quality.ranking_eval import interval

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"
ARMS = (("uniform", "Uniform replay", "#34495e"),
        ("quality", "Outcome gate", "#95a5a6"),
        ("quality_return", "Return gate", "#d35400"),
        ("quality_offline", "Offline gate", "#27ae60"))


def load(arm, seed):
    with (OUT / f"eval_{arm}_seed{seed}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot(metric, ylabel, name, bounds=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for arm, label, color in ARMS:
        runs = [load(arm, seed) for seed in range(5)]
        steps = np.asarray([int(row["env_steps"]) for row in runs[0]])
        if len(steps) != 11 or any([int(row["env_steps"]) for row in run] != list(steps)
                                   for run in runs):
            raise RuntimeError(f"Mismatched evaluation schedule for {arm}")
        values = np.asarray([[float(row[metric]) for row in run] for run in runs])
        summary = np.asarray([interval(values[:, index]) for index in range(len(steps))])
        mean, lo, hi = summary.T
        if bounds is not None:
            lo = np.clip(lo, *bounds)
            hi = np.clip(hi, *bounds)
        ax.plot(steps / 1000, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps / 1000, lo, hi, color=color, alpha=.16)
    ax.set_xlabel("Online training environment steps (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    if bounds is not None:
        ax.set_ylim(*bounds)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)


def main():
    plot("success_rate", "Fixed evaluation success rate", "success_vs_steps.png", (0, 1))
    plot("mean_return", "Fixed evaluation mean episode return", "reward_vs_steps.png")
    plot("weighted_active", "Fraction of seeds using quality weights", "weighted_activation.png", (0, 1))
    print("Wrote three five-seed pilot figures")


if __name__ == "__main__":
    main()
