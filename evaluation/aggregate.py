"""Aggregate per-seed evaluations without inventing unreached thresholds.

Input columns: variant,seed,env_steps,success_rate,mean_return,eval_episodes,
eval_env_steps,wall_time_s. The first step count is training only; the second
counts the cumulative evaluation interactions separately.
Outputs summary.json, success.png, reward.png.
"""

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


THRESHOLDS = (0.5, 0.8, 0.9)
VARIANT_LABELS = {
    "A": "A: SAC baseline",
    "B": "B: privileged critic SAC",
    "C": "C: privileged policy SAC",
}


def load_rows(path):
    paths = sorted(path.glob("eval_*.csv")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No evaluation CSV files found at {path}")
    rows = []
    for item in paths:
        with open(item, newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    required = {"variant", "seed", "env_steps", "success_rate", "mean_return", "eval_episodes"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected columns {sorted(required)} and at least one row")
    result = []
    for row in rows:
        parsed = {
            "variant": row["variant"], "seed": int(row["seed"]),
            "env_steps": int(row["env_steps"]),
            "success_rate": float(row["success_rate"]),
            "mean_return": float(row["mean_return"]),
            "eval_episodes": int(row["eval_episodes"]),
            "eval_env_steps": int(row.get("eval_env_steps") or 0),
            "wall_time_s": float(row.get("wall_time_s") or 0),
        }
        if not 0 <= parsed["success_rate"] <= 1 or parsed["eval_episodes"] <= 0:
            raise ValueError(f"Invalid success or episode count: {row}")
        result.append(parsed)
    return result


def exact_sign_flip_p(differences):
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return None
    observed = abs(differences.mean())
    means = [abs(np.mean(differences * signs)) for signs in itertools.product((-1, 1), repeat=len(differences))]
    return sum(value >= observed - 1e-12 for value in means) / len(means)


def summarize(rows, conditioning_interactions=0):
    curves = defaultdict(list)
    for row in rows:
        curves[(row["variant"], row["seed"])].append(row)
    for key, curve in curves.items():
        curve.sort(key=lambda x: x["env_steps"])
        steps = [row["env_steps"] for row in curve]
        if len(steps) != len(set(steps)):
            raise ValueError(f"Duplicate checkpoint in {key}")
    by_variant = defaultdict(dict)
    for (variant, seed), curve in curves.items():
        by_variant[variant][seed] = curve
    result = {"variants": {}, "paired_comparisons": {},
              "shared_conditioning_interactions": conditioning_interactions}
    for variant, seed_curves in by_variant.items():
        per_seed = {}
        for seed, curve in seed_curves.items():
            x = np.array([row["env_steps"] for row in curve], dtype=float)
            y = np.array([row["success_rate"] for row in curve], dtype=float)
            reached = {str(t): next((int(x[i]) for i in range(len(x)) if y[i] >= t), None) for t in THRESHOLDS}
            retained = {
                str(t): next(
                    (int(x[i]) for i in range(len(x) - 1) if np.all(y[i:] >= t)), None
                ) for t in THRESHOLDS
            }
            total_reached = {str(t): next((int(x[i] + curve[i]["eval_env_steps"]) for i in range(len(x)) if y[i] >= t), None) for t in THRESHOLDS}
            full_reached = {
                key: value + conditioning_interactions if value is not None else None
                for key, value in total_reached.items()
            }
            wall_reached = {str(t): next((float(curve[i]["wall_time_s"]) for i in range(len(x)) if y[i] >= t), None) for t in THRESHOLDS}
            auc = float(np.trapz(y, x) / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else None
            per_seed[str(seed)] = {
                "threshold_steps": reached,
                "retained_threshold_steps": retained,
                "threshold_total_interactions_including_evaluation": total_reached,
                "threshold_total_interactions_including_conditioning_and_evaluation": full_reached,
                "threshold_wall_time_s": wall_reached,
                "success_auc": auc,
                "peak_success_rate": float(y.max()),
                "peak_at_train_steps": int(x[int(y.argmax())]),
                "last_env_step": int(x[-1]), "checkpoints": len(x),
                "last_eval_env_steps": curve[-1]["eval_env_steps"],
                "last_wall_time_s": curve[-1]["wall_time_s"],
            }
        result["variants"][variant] = {"seeds": per_seed, "count": len(seed_curves)}
    for baseline, contender in (("A", "B"), ("A", "C"), ("B", "C")):
        if baseline not in by_variant or contender not in by_variant:
            continue
        paired = []
        for seed in sorted(set(by_variant[baseline]) & set(by_variant[contender])):
            a_curve, b_curve = by_variant[baseline][seed], by_variant[contender][seed]
            a_steps = [row["env_steps"] for row in a_curve]
            b_steps = [row["env_steps"] for row in b_curve]
            if a_steps != b_steps:
                continue
            a = result["variants"][baseline]["seeds"][str(seed)]["success_auc"]
            b = result["variants"][contender]["seeds"][str(seed)]["success_auc"]
            if a is not None and b is not None:
                paired.append((seed, b - a))
        if paired:
            diffs = [gain for _, gain in paired]
            rng = np.random.default_rng(20260929)
            boots = np.mean(
                rng.choice(np.asarray(diffs), size=(10000, len(diffs)), replace=True), axis=1
            )
            result["paired_comparisons"][f"{contender}_minus_{baseline}"] = {
                "shared_seeds_with_same_checkpoints": [seed for seed, _ in paired],
                "per_seed_success_auc_gain": {str(seed): gain for seed, gain in paired},
                "mean_success_auc_gain": float(np.mean(diffs)),
                "paired_bootstrap_95pct_ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                "exact_two_sided_sign_flip_p": exact_sign_flip_p(diffs),
                "n": len(diffs),
            }
    return result


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["env_steps"])].append(row)
    for metric, name, ylabel in (
        ("success_rate", "success.png", "Success rate"),
        ("mean_return", "reward.png", "Mean episode return"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for variant in sorted({row["variant"] for row in rows}):
            steps = sorted(step for v, step in grouped if v == variant)
            mean = [np.mean([row[metric] for row in grouped[(variant, step)]]) for step in steps]
            sem = [np.std([row[metric] for row in grouped[(variant, step)]], ddof=1) / math.sqrt(len(grouped[(variant, step)])) if len(grouped[(variant, step)]) > 1 else 0 for step in steps]
            ax.plot(steps, mean, label=VARIANT_LABELS.get(variant, variant))
            ax.fill_between(steps, np.array(mean) - sem, np.array(mean) + sem, alpha=0.2)
        ax.set(xlabel="Online training environment steps", ylabel=ylabel)
        if metric == "success_rate":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)


def write_tables(rows, summary, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["env_steps"])].append(row)
    with (output_dir / "aggregate_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "env_steps", "seeds", "mean_success_rate", "sem_success_rate", "mean_return", "sem_return"))
        for (variant, step), group in sorted(grouped.items()):
            success = np.array([item["success_rate"] for item in group])
            returns = np.array([item["mean_return"] for item in group])
            sem = lambda values: float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            writer.writerow((variant, step, len(group), float(success.mean()), sem(success), float(returns.mean()), sem(returns)))
    with (output_dir / "thresholds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "seed", "success_threshold", "train_env_steps",
                         "retained_from_train_steps",
                         "total_interactions_including_evaluation",
                         "total_interactions_including_conditioning_and_evaluation", "wall_time_s"))
        for variant, variant_data in sorted(summary["variants"].items()):
            for seed, seed_data in sorted(variant_data["seeds"].items(), key=lambda item: int(item[0])):
                for threshold in THRESHOLDS:
                    key = str(threshold)
                    writer.writerow((
                        variant, seed, threshold,
                        seed_data["threshold_steps"][key],
                        seed_data["retained_threshold_steps"][key],
                        seed_data["threshold_total_interactions_including_evaluation"][key],
                        seed_data["threshold_total_interactions_including_conditioning_and_evaluation"][key],
                        seed_data["threshold_wall_time_s"][key],
                    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="One evaluation CSV or a directory of eval_*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--conditioning-interactions", type=int, default=0,
                        help="Shared expert/DAgger simulator steps before online SAC")
    args = parser.parse_args()
    if args.conditioning_interactions < 0:
        raise ValueError("conditioning-interactions must be nonnegative")
    rows = load_rows(args.csv_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, args.conditioning_interactions)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows, args.output_dir)
    write_tables(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
