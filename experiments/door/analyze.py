"""Audit complete Door runs, calculate paired statistics and draw curves."""

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/door"
VARIANTS = ("A", "B", "C", "D")
SEEDS = tuple(range(5))
BUDGET = 500000
COLORS = {"A": "#666666", "B": "#1769aa", "C": "#2e8b57", "D": "#d97706"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def paired_interval(x, y, rng, iterations=20000):
    d = np.asarray(x) - np.asarray(y)
    boot = d[rng.integers(0, len(d), size=(iterations, len(d)))].mean(axis=1)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(d))))
    null = (signs * d[None, :]).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12))
    return float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), p


def first_cross(rows, threshold):
    for row in rows:
        if float(row["success_rate"]) >= threshold:
            return int(row["env_steps"])
    return None


def main():
    curves = {}
    seed_rows = []
    per_seed = {}
    for variant in VARIANTS:
        curves[variant] = []
        for seed in SEEDS:
            path = RESULTS / f"eval_{variant}_seed{seed}.csv"
            if not path.exists():
                raise RuntimeError(f"Missing experiment: {path}")
            rows = read_csv(path)
            steps = np.asarray([int(r["env_steps"]) for r in rows])
            success = np.asarray([float(r["success_rate"]) for r in rows])
            reward = np.asarray([float(r["mean_return"]) for r in rows])
            contact = np.asarray([float(r["contact_rate"]) for r in rows])
            if len(rows) != 21 or steps[0] != 0 or steps[-1] != BUDGET:
                raise RuntimeError(f"Incomplete experiment: {path} {steps}")
            if not np.all(np.diff(steps) > 0) or int(rows[-1]["eval_episodes"]) != 64:
                raise RuntimeError(f"Invalid evaluation grid: {path}")
            entry = {
                "auc_success": float(np.trapz(success, steps) / BUDGET),
                "auc_reward": float(np.trapz(reward, steps) / BUDGET),
                "final_success": float(success[-1]), "best_success": float(success.max()),
                "final_reward": float(reward[-1]),
                "first_50": first_cross(rows, 0.5),
                "first_80": first_cross(rows, 0.8),
                "first_90": first_cross(rows, 0.9),
                "eval_env_steps": int(rows[-1]["eval_env_steps"]),
            }
            per_seed[(variant, seed)] = entry
            seed_rows.append((variant, seed) + tuple("" if v is None else v for v in entry.values()))
            curves[variant].append((steps, success, reward, contact))
        for i in range(1, len(SEEDS)):
            if not np.array_equal(curves[variant][i][0], curves[variant][0][0]):
                raise RuntimeError(f"Misaligned evaluation grids for {variant}")

    header = ("variant", "seed", "auc_success", "auc_reward", "final_success", "best_success",
              "final_reward", "first_50", "first_80", "first_90", "eval_env_steps")
    write_csv(RESULTS / "seed_metrics.csv", header, seed_rows)
    aggregate = []
    summary = {}
    for variant in VARIANTS:
        steps = curves[variant][0][0]
        success = np.stack([x[1] for x in curves[variant]])
        reward = np.stack([x[2] for x in curves[variant]])
        contact = np.stack([x[3] for x in curves[variant]])
        for i, step in enumerate(steps):
            aggregate.append((variant, int(step), float(success[:, i].mean()), float(success[:, i].std(ddof=1)),
                              float(reward[:, i].mean()), float(reward[:, i].std(ddof=1)),
                              float(contact[:, i].mean()), float(contact[:, i].std(ddof=1))))
        summary[variant] = {
            "mean_auc_success": float(np.mean([per_seed[(variant, s)]["auc_success"] for s in SEEDS])),
            "sd_auc_success": float(np.std([per_seed[(variant, s)]["auc_success"] for s in SEEDS], ddof=1)),
            "mean_final_success": float(success[:, -1].mean()),
            "sd_final_success": float(success[:, -1].std(ddof=1)),
            "mean_best_success": float(np.mean([per_seed[(variant, s)]["best_success"] for s in SEEDS])),
            "threshold_steps": {
                str(int(t * 100)): [per_seed[(variant, s)][f"first_{int(t*100)}"] for s in SEEDS]
                for t in (0.5, 0.8, 0.9)
            },
        }
    summary["protocol_totals"] = {
        "online_training_interactions": len(VARIANTS) * len(SEEDS) * BUDGET,
        "periodic_evaluation_interactions": sum(
            per_seed[(variant, seed)]["eval_env_steps"]
            for variant in VARIANTS for seed in SEEDS),
        "shared_expert_collection_interactions": 277184,
        "evaluation_episodes_per_checkpoint": 64,
        "checkpoint_count_per_run": 21,
    }
    write_csv(RESULTS / "aggregate_curves.csv",
              ("variant", "env_steps", "success_mean", "success_sd", "reward_mean", "reward_sd",
               "contact_mean", "contact_sd"), aggregate)

    rng = np.random.default_rng(20260929)
    paired = []
    for left, right in (("B", "A"), ("D", "B"), ("C", "A"), ("C", "B")):
        for metric in ("auc_success", "final_success", "auc_reward"):
            x = [per_seed[(left, seed)][metric] for seed in SEEDS]
            y = [per_seed[(right, seed)][metric] for seed in SEEDS]
            mean, lo, hi, p = paired_interval(x, y, rng)
            paired.append((f"{left}-{right}", metric, mean, lo, hi, p, len(SEEDS)))
    write_csv(RESULTS / "paired_effects.csv",
              ("contrast", "metric", "mean_difference", "bootstrap_ci_2p5", "bootstrap_ci_97p5",
               "exact_signflip_p_two_sided", "seed_pairs"), paired)
    summary["paired_effects"] = [dict(zip(("contrast", "metric", "mean_difference", "ci_low", "ci_high", "p", "seed_pairs"), row)) for row in paired]

    def plot_curve(index, name, ylabel, filename, ylim=None):
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        for variant in VARIANTS:
            steps = curves[variant][0][0]
            values = np.stack([r[index] for r in curves[variant]])
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            for seed_values in values:
                ax.plot(steps / 1000, seed_values, color=COLORS[variant], alpha=0.10, linewidth=0.8)
            ax.plot(steps / 1000, mean, color=COLORS[variant], linewidth=2.3,
                    label=f"{variant}: {name}")
            ax.fill_between(steps / 1000, mean - sd, mean + sd, color=COLORS[variant], alpha=0.12)
        ax.set_xlabel("Online environment steps (thousands)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, BUDGET / 1000)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(RESULTS / filename, dpi=180)
        plt.close(fig)

    plot_curve(1, "success", "Success rate (64 evaluation episodes/seed)", "success_curve.png", (0, 1.02))
    plot_curve(2, "reward", "Mean episode return", "reward_curve.png")
    plot_curve(3, "contact", "Both-fingers contact rate", "contact_curve.png", (0, 1.02))

    replay_rows = []
    for seed in SEEDS:
        path = RESULTS / f"replay_D_seed{seed}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing D replay diagnostics: {path}")
        for row in read_csv(path):
            step = int(row["env_steps"])
            replay_rows.append((seed, step,
                float(row["effective_sample_size"]) / step,
                float(row["top_decile_mass"]),
                float(row["success_sampling_mass"]),
                float(row["success_uniform_fraction"]),
                int(row["completed_trajectories"])))
    write_csv(RESULTS / "replay_distribution.csv",
              ("seed", "env_steps", "ess_fraction", "top_decile_mass",
               "success_sampling_mass", "success_uniform_fraction", "completed_trajectories"), replay_rows)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for index, label, color in ((3, "Top 10% transition probability mass", "#d97706"),
                                (4, "Success transition probability mass", "#2e8b57"),
                                (5, "Success fraction under uniform replay", "#666666")):
        matrix = np.stack([np.asarray([r[index] for r in replay_rows if r[0] == seed]) for seed in SEEDS])
        steps = np.asarray([r[1] for r in replay_rows if r[0] == SEEDS[0]])
        ax.plot(steps / 1000, matrix.mean(axis=0), label=label, color=color, linewidth=2)
        ax.fill_between(steps / 1000, matrix.mean(axis=0) - matrix.std(axis=0, ddof=1),
                        matrix.mean(axis=0) + matrix.std(axis=0, ddof=1), color=color, alpha=0.12)
    ax.set_xlabel("Online environment steps (thousands)")
    ax.set_ylabel("Sampling probability mass")
    ax.set_xlim(0, BUDGET / 1000)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "replay_distribution.png", dpi=180)
    plt.close(fig)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
