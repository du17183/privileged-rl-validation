"""Counterfactual replay sampling audit; does not claim an RL policy effect."""

import csv
from pathlib import Path

import numpy as np

from replay.quality_weighted_replay import TrajectoryReplay
from trajectory_quality.ranking_eval import interval, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"


def score_lookup(path, identity):
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = identity(row)
            result.setdefault(key, {})[row["trajectory"]] = float(row["score"])
    return result


def main():
    episode = np.load(OUT / "episode_features.npz")
    offline = score_lookup(OUT / "offline_model_scores.csv",
                           lambda row: (row["test_split"], row["model"], int(row["model_seed"])))
    q = score_lookup(OUT / "q_baseline_scores.csv",
                     lambda row: (row["test_split"], row["model"], row["checkpoint_mode"],
                                  int(row["critic_seed"])))
    models = ("E2_best_Q", "Q0_best_Q", "MC", "EARLY50", "EARLY100", "QUALITY")
    rows = []
    for split in ("test", "stress"):
        mask = episode["split"] == split
        keys = episode["key"][mask]
        sources = episode["source"][mask]
        success = episode["success"][mask].astype(float)
        returns = episode["return_undiscounted"][mask].astype(float)
        lengths = episode["length"][mask].astype(float)
        if len(keys) != 1000:
            raise RuntimeError("Locked held-out set changed")
        uniform = np.full(len(keys), 1 / len(keys))
        for model in models:
            for seed in range(5):
                lookup = (q[(split, model.split("_")[0], "best", seed)]
                          if model.endswith("_best_Q") else offline[(split, model, seed)])
                scores = np.asarray([lookup[key] for key in keys], dtype=float)
                buffer = TrajectoryReplay(keys, sources, scores, scores)
                probabilities = buffer.probabilities("value" if model.endswith("_best_Q") else "quality")
                transition_uniform = lengths / lengths.sum()
                transition_weighted = lengths * probabilities
                transition_weighted /= transition_weighted.sum()
                baseline = float(uniform @ success)
                source_diff = max(abs(float(probabilities[sources == source].sum()) -
                                      float(uniform[sources == source].sum()))
                                  for source in np.unique(sources))
                rows.append({"test_split": split, "model": model, "model_seed": seed,
                             "uniform_success": baseline,
                             "weighted_success": float(probabilities @ success),
                             "success_enrichment": float((probabilities - uniform) @ success),
                             "uniform_return": float(uniform @ returns),
                             "weighted_return": float(probabilities @ returns),
                             "return_enrichment": float((probabilities - uniform) @ returns),
                             "uniform_transition_success": float(transition_uniform @ success),
                             "weighted_transition_success": float(transition_weighted @ success),
                             "transition_success_enrichment": float(
                                 (transition_weighted - transition_uniform) @ success),
                             "effective_sample_size": float(1 / (probabilities @ probabilities)),
                             "max_source_mass_difference": source_diff})
    write_csv(OUT / "offline_replay_metrics.csv", rows)
    aggregate = []
    for split in ("test", "stress"):
        for model in models:
            subset = [row for row in rows if row["test_split"] == split and row["model"] == model]
            for metric in ("weighted_success", "success_enrichment", "weighted_return",
                           "return_enrichment", "weighted_transition_success",
                           "transition_success_enrichment", "effective_sample_size",
                           "max_source_mass_difference"):
                mean, lo, hi = interval([row[metric] for row in subset])
                aggregate.append({"test_split": split, "model": model, "metric": metric,
                                  "mean": mean, "ci95_low": lo, "ci95_high": hi})
    write_csv(OUT / "offline_replay_aggregate.csv", aggregate)
    for row in aggregate:
        if row["metric"] == "success_enrichment":
            print(row["test_split"], row["model"],
                  f"success enrichment={row['mean']:.3f} "
                  f"[{row['ci95_low']:.3f},{row['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
