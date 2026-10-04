"""Publication-style Phase 7 success and drift figures from completed CSVs."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def rows(arm, seed, kind="eval"):
    if arm in ("E0", "L000"):
        old = "B3" if arm == "E0" else "B1"
        path = ROOT / "results" / "phase6_stable_online_rl" / f"{kind}_P6{old}_seed{seed}.csv"
    else:
        path = OUT / f"{kind}_P7{arm}_seed{seed}.csv"
    with path.open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if int(r["env_steps"]) <= 300000]


def draw(arms, metric, kind, filename, ylabel):
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    palette = plt.get_cmap("tab10")
    for j, arm in enumerate(arms):
        curves = [rows(arm, seed, kind) for seed in range(5)]
        x = np.asarray([int(r["env_steps"]) for r in curves[0]]) / 1000
        values = np.asarray([[float(r[metric]) for r in run] for run in curves])
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(5)
        color = palette(j)
        ax.plot(x, mean, label=arm, color=color, lw=1.8)
        ax.fill_between(x, mean - 2.77645 * sem, mean + 2.77645 * sem,
                        color=color, alpha=0.13)
    ax.set(xlabel="Training environment steps (thousands)", ylabel=ylabel,
           xlim=(0, 300))
    if metric == "success_rate":
        ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(ncol=min(5, len(arms)), frameon=False)
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def main():
    draw(("E0", "E1L", "E2L", "E3", "G1"), "success_rate", "eval",
         "exploration_guard_success.png", "Deterministic success rate")
    draw(("E0", "E1", "E2"), "success_rate", "eval",
         "exploratory_target_entropy_success.png", "Deterministic success rate")
    draw(("L000", "L001", "L005", "L010", "L050", "E0"), "success_rate", "eval",
         "bc_weight_success.png", "Deterministic success rate")
    draw(("E0", "E1L", "E2L", "E3", "G1"), "policy_entropy", "diagnostics",
         "policy_entropy.png", "Estimated policy entropy on fixed expert observations")
    draw(("E0", "E1L", "E2L", "E3", "G1"), "action_drift_mse", "diagnostics",
         "action_drift.png", "Action mean drift MSE from BC start")
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    for arm, field, label, color in (("E0", "success_rate", "E0 no guard", "#555555"),
                                     ("G1", "raw_success_rate", "G1 before guard", "#d95f02"),
                                     ("G1", "success_rate", "G1 deployed", "#1b9e77")):
        curves = [rows(arm, seed) for seed in range(5)]
        x = np.asarray([int(r["env_steps"]) for r in curves[0]]) / 1000
        values = np.asarray([[float(r[field]) for r in run] for run in curves])
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(5)
        ax.plot(x, mean, label=label, color=color, lw=1.8)
        ax.fill_between(x, mean - 2.77645 * sem, mean + 2.77645 * sem,
                        color=color, alpha=0.12)
    ax.set(xlabel="Training environment steps (thousands)",
           ylabel="Deterministic success rate", xlim=(0, 300), ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "guard_raw_vs_deployed.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
