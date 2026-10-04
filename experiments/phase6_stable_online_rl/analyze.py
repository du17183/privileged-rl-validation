"""Recompute paired Phase 6 statistics and static scientific figures.

Refuses to publish an aggregate until all 30 500k/64-episode runs exist.
"""

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase6_stable_online_rl"
VARIANTS = ("A0", "A1", "A2", "B1", "B2", "B3")
SEEDS = range(5)
T95_N5 = 2.7764451051977987
CONTRASTS = (("A1", "A0"), ("A2", "A0"), ("A1", "A2"),
             ("B1", "A2"), ("B2", "B1"), ("B3", "B1"),
             ("B3", "A0"), ("B3", "A2"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Exactly five finite seed values required")
    mean = float(values.mean())
    radius = T95_N5 * float(values.std(ddof=1)) / np.sqrt(5)
    return mean, mean-radius, mean+radius


def signflip_p(differences):
    values = np.asarray(differences, dtype=float)
    observed = abs(values.mean())
    means = [abs(np.mean(values * np.asarray(signs)))
             for signs in itertools.product((-1, 1), repeat=len(values))]
    return float(np.mean(np.asarray(means) >= observed-1e-12))


def load():
    data = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            run = f"P6{variant}_seed{seed}"
            path = OUT / f"eval_{run}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            eval_rows = rows(path)
            steps = np.asarray([int(r["env_steps"]) for r in eval_rows])
            if steps[0] != 0 or steps[-1] != 500000 or len(steps) != 51:
                raise ValueError(f"Incomplete 500k/10k run: {run}, rows={len(steps)}, last={steps[-1]}")
            episodes = np.asarray([int(r["eval_episodes"]) for r in eval_rows])
            if np.any(episodes < 50):
                raise ValueError(f"Evaluation has fewer than 50 episodes: {run}")
            success = np.asarray([float(r["success_rate"]) for r in eval_rows])
            reward = np.asarray([float(r["mean_return"]) for r in eval_rows])
            contact = np.asarray([float(r["contact_rate"]) for r in eval_rows])
            diag = rows(OUT / f"diagnostics_{run}.csv")
            if len(diag) != 50:
                raise ValueError(f"Incomplete diagnostics: {run}")
            data[(variant, seed)] = {
                "steps": steps, "success": success, "reward": reward,
                "contact": contact,
                "eval_env_steps": int(eval_rows[-1]["eval_env_steps"]),
                "auc": float(np.trapz(success, steps)/500000),
                "reward_auc": float(np.trapz(reward, steps)/500000),
                "final": float(success[-1]), "best": float(success.max()),
                "best_step": int(steps[np.argmax(success)]),
                "degradation": float(success.max()-success[-1]),
                "initial": float(success[0]), "diagnostics": diag,
                "first_eval": float(success[1]),
                "initial_contact": float(contact[0]),
                "first_eval_contact": float(contact[1]),
                "final_contact": float(contact[-1]),
                "train_success_count": int(sum(
                    round(float(r["train_success_rate"])*int(r["train_episodes"]))
                    for r in eval_rows[1:] if r["train_success_rate"] not in ("", "nan"))),
                "train_episode_count": int(sum(int(r["train_episodes"]) for r in eval_rows[1:])),
            }
    return data


def write_csv(path, header, content):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(content)


def plot_curves(data, field, filename, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in VARIANTS:
        series = np.stack([data[(variant, seed)][field] for seed in SEEDS])
        steps = data[(variant, 0)]["steps"]
        mean = series.mean(axis=0)
        radius = T95_N5 * series.std(axis=0, ddof=1)/np.sqrt(5)
        line, = ax.plot(steps, mean, label=variant)
        low, high = mean-radius, mean+radius
        if field in ("success", "contact"):
            low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
        ax.fill_between(steps, low, high,
                        alpha=0.12, color=line.get_color())
    ax.set(xlabel="Training environment steps", ylabel=ylabel)
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def main():
    data = load()
    per_seed = []
    aggregate = []
    for variant in VARIANTS:
        for seed in SEEDS:
            d = data[(variant, seed)]
            per_seed.append((variant, seed, *(d[key] for key in (
                "auc", "reward_auc", "initial", "best", "best_step", "final",
                "degradation", "first_eval", "initial_contact",
                "first_eval_contact", "final_contact",
                "train_success_count", "train_episode_count"))))
        for metric in ("auc", "reward_auc", "initial", "first_eval", "best",
                       "final", "degradation", "initial_contact",
                       "first_eval_contact", "final_contact"):
            mean, lo, hi = summarize([data[(variant, s)][metric] for s in SEEDS])
            if metric != "reward_auc":
                lo, hi = max(0.0, lo), min(1.0, hi)
            aggregate.append((variant, metric, mean, lo, hi))
    write_csv(OUT / "per_seed.csv",
              ("variant", "seed", "auc", "reward_auc", "initial_success",
               "best_success", "best_step", "final_success", "degradation",
               "first_eval_success", "initial_contact", "first_eval_contact",
               "final_contact", "train_success_count",
               "train_episode_count"), per_seed)
    write_csv(OUT / "aggregate.csv",
              ("variant", "metric", "mean", "ci95_low", "ci95_high"), aggregate)
    contrasts = []
    for treatment, control in CONTRASTS:
        for metric in ("auc", "final", "best", "degradation"):
            differences = [data[(treatment, s)][metric]-data[(control, s)][metric]
                           for s in SEEDS]
            mean, lo, hi = summarize(differences)
            lo, hi = max(-1.0, lo), min(1.0, hi)
            contrasts.append((treatment, control, metric, mean, lo, hi,
                              signflip_p(differences)))
    write_csv(OUT / "paired_comparisons.csv",
              ("treatment", "control", "metric", "mean_difference",
               "ci95_low", "ci95_high", "exact_two_sided_signflip_p"), contrasts)
    degradation = []
    for variant in VARIANTS:
        for seed in SEEDS:
            d = data[(variant, seed)]
            success = d["success"]
            best_index = int(np.argmax(success))
            onset = None
            for index in range(best_index+1, len(success)-1):
                if (success[index] <= success[best_index]-0.25 and
                        success[index+1] <= success[best_index]-0.25):
                    onset = index
                    break
            diagnostic = d["diagnostics"]
            last = diagnostic[-1]
            peak = diagnostic[best_index-1] if best_index else None
            before_drop = diagnostic[onset-2] if onset is not None and onset >= 2 else None
            at_drop = diagnostic[onset-1] if onset is not None else None
            degradation.append((variant, seed, int(d["steps"][best_index]),
                                int(d["steps"][onset]) if onset else "",
                                float(success[best_index]), float(success[-1]),
                                float(peak["q_mean"]) if peak else "", float(last["q_mean"]),
                                float(peak["q_variance"]) if peak else "", float(last["q_variance"]),
                                float(peak["action_drift_mse"]) if peak else "",
                                float(last["action_drift_mse"]),
                                float(before_drop["q_mean"]) if before_drop else "",
                                float(at_drop["q_mean"]) if at_drop else "",
                                float(before_drop["q_variance"]) if before_drop else "",
                                float(at_drop["q_variance"]) if at_drop else "",
                                float(before_drop["action_drift_mse"]) if before_drop else "",
                                float(at_drop["action_drift_mse"]) if at_drop else "",
                                d["train_success_count"], d["train_episode_count"]))
    write_csv(OUT / "degradation_diagnostics.csv",
              ("variant", "seed", "selected_best_step", "sustained_drop_onset_step",
               "selected_best_success", "final_success", "q_mean_at_best",
               "q_mean_final", "q_variance_at_best", "q_variance_final",
               "policy_drift_at_best", "policy_drift_final",
               "q_mean_before_drop", "q_mean_at_drop", "q_variance_before_drop",
               "q_variance_at_drop", "policy_drift_before_drop",
               "policy_drift_at_drop",
               "online_training_successes", "online_training_episodes"),
              degradation)
    heldout = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for mode in ("best", "final"):
                path = OUT / f"heldout_P6{variant}_seed{seed}_{mode}.csv"
                if path.exists():
                    row = rows(path)[0]
                    heldout.append((variant, seed, mode, int(row["selected_steps"]),
                                    float(row["heldout_success_rate"]),
                                    float(row["heldout_mean_return"])))
    write_csv(OUT / "heldout_per_seed.csv",
              ("variant", "seed", "mode", "selected_steps",
               "success_rate", "mean_return"), heldout)
    if len(heldout) == 60:
        lookup = {(r[0], r[1], r[2]): r[4] for r in heldout}
        heldout_aggregate = []
        for variant in VARIANTS:
            for mode in ("best", "final"):
                values = [r[4] for r in heldout if r[0] == variant and r[2] == mode]
                mean, lo, hi = summarize(values)
                heldout_aggregate.append((variant, mode, mean, max(0.0, lo), min(1.0, hi)))
            differences = [lookup[(variant, seed, "best")] -
                           lookup[(variant, seed, "final")] for seed in SEEDS]
            mean, lo, hi = summarize(differences)
            heldout_aggregate.append((variant, "best_minus_final", mean,
                                      max(-1.0, lo), min(1.0, hi)))
        write_csv(OUT / "heldout_aggregate.csv",
                  ("variant", "mode", "mean", "ci95_low", "ci95_high"),
                  heldout_aggregate)
        heldout_paired = []
        for treatment, control in CONTRASTS:
            for mode in ("best", "final"):
                differences = [lookup[(treatment, seed, mode)] -
                               lookup[(control, seed, mode)] for seed in SEEDS]
                mean, lo, hi = summarize(differences)
                heldout_paired.append((treatment, control, mode,
                                       mean, max(-1.0, lo), min(1.0, hi),
                                       signflip_p(differences)))
        write_csv(OUT / "heldout_paired_comparisons.csv",
                  ("treatment", "control", "mode", "mean_difference",
                   "ci95_low", "ci95_high", "exact_two_sided_signflip_p"),
                  heldout_paired)
        exploration = []
        for variant in ("B1", "B3"):
            for seed in SEEDS:
                path = OUT / f"heldout_P6{variant}_seed{seed}_best_stochastic.csv"
                if path.exists():
                    eval_row = rows(path)[0]
                    stochastic = float(eval_row["heldout_success_rate"])
                    deterministic = lookup[(variant, seed, "best")]
                    exploration.append((variant, seed, deterministic, stochastic,
                                        deterministic-stochastic,
                                        float(eval_row["reset_mean_action_std"])))
        write_csv(OUT / "exploration_gap_per_seed.csv",
                  ("variant", "seed", "deterministic_best_success",
                   "stochastic_best_success", "gap", "reset_mean_action_std"),
                  exploration)
        if len(exploration) == 10:
            write_csv(OUT / "exploration_gap_aggregate.csv",
                      ("variant", "mean_gap", "ci95_low", "ci95_high"),
                      ((variant, *summarize([r[4] for r in exploration if r[0] == variant]))
                       for variant in ("B1", "B3")))
    plot_curves(data, "success", "success_vs_steps.png", "Fixed-evaluation success rate")
    plot_curves(data, "reward", "reward_vs_steps.png", "Fixed-evaluation mean return")
    plot_curves(data, "contact", "contact_vs_steps.png", "Fixed-evaluation contact rate")
    (OUT / "analysis_manifest.json").write_text(json.dumps({
        "variants": VARIANTS, "seeds": list(SEEDS), "training_steps_per_run": 500000,
        "total_training_env_steps": 15000000,
        "total_fixed_evaluation_env_steps": sum(d["eval_env_steps"] for d in data.values()),
        "evaluation_episodes_per_checkpoint": 64,
        "B0_equivalent": "A2", "complete_runs": len(data),
        "heldout_best_final_records": len(heldout),
        "ci": "two-sided Student-t across five paired seeds",
        "ci_bounds": "rate intervals clipped to [0,1]; paired rate differences clipped to [-1,1]",
        "test": "exact two-sided sign-flip over all 2^5 sign vectors",
    }, indent=2), encoding="utf-8")
    print(f"Analyzed {len(data)} complete runs and {len(heldout)} heldout records")


if __name__ == "__main__":
    main()
