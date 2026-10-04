"""Locked primary and stress tests for offline value and trajectory models."""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from algorithms.asymmetric_sac import TwinQ, mlp
from trajectory_quality.quality_model import TrajectoryQualityModel
from value_analysis.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"
CKPT = ROOT / "checkpoints" / "value_calibration"
METHODS = ("MC", "TD0", "TDMC", "EARLY50", "EARLY100", "QUALITY")


@torch.no_grad()
def predict(method, seed, episode, mask):
    saved = torch.load(CKPT / f"{method}_seed{seed}.pt", map_location="cpu", weights_only=False)
    if method in ("EARLY50", "EARLY100", "QUALITY"):
        model = TrajectoryQualityModel()
        model.load_state_dict(saved["model"])
        model.eval()
        feature_key = {"EARLY50": "early50", "EARLY100": "early100",
                       "QUALITY": "sequence"}[method]
        sequence = (episode[feature_key][mask].astype(np.float32) -
                    saved["input_center"]) / saved["input_scale"]
        scores = []
        for start in range(0, len(sequence), 128):
            logit, _ = model(torch.from_numpy(sequence[start:start + 128]))
            scores.extend(torch.sigmoid(logit).flatten().numpy())
        return np.asarray(scores, dtype=float), saved
    robot = torch.from_numpy(episode["start_robot"][mask].astype(np.float32))
    action = torch.from_numpy(episode["start_action"][mask].astype(np.float32))
    if method == "MC":
        model = mlp(33, 1)
        model.load_state_dict(saved["model"])
        model.eval()
        score = model(torch.cat((robot, action), dim=1)).flatten()
        score = score * saved["mc_scale"] + saved["mc_mean"]
    else:
        model = TwinQ(26, 7)
        model.load_state_dict(saved["model"])
        model.eval()
        score = torch.minimum(*model(robot, action)).flatten()
    return score.numpy().astype(float), saved


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - radius, mean + radius


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    episode = np.load(OUT / "episode_features.npz")
    rows, predictions = [], []
    for split in ("test", "stress"):
        mask = episode["split"] == split
        if mask.sum() != 1000:
            raise RuntimeError(f"Expected 1000 {split} episodes, got {mask.sum()}")
        truth = {name: episode[name][mask] for name in
                 ("return_discounted", "return_undiscounted", "success", "source", "key")}
        for method in METHODS:
            for seed in range(5):
                score, saved = predict(method, seed, episode, mask)
                stats = evaluate(score, truth["return_discounted"],
                                 truth["return_undiscounted"], truth["success"], truth["source"])
                rows.append({"test_split": split, "model": method, "model_seed": seed,
                             "best_iteration": saved.get("best_update", saved.get("best_epoch")),
                             **stats})
                predictions.extend({"test_split": split, "model": method,
                                    "model_seed": seed, "trajectory": str(key),
                                    "score": float(value)}
                                   for key, value in zip(truth["key"], score))
        for method, score in (("REALIZED_RETURN_ORACLE", truth["return_discounted"]),
                              ("SUCCESS_ORACLE", truth["success"])):
            stats = evaluate(score, truth["return_discounted"],
                             truth["return_undiscounted"], truth["success"], truth["source"])
            rows.append({"test_split": split, "model": method, "model_seed": -1,
                         "best_iteration": -1, **stats})
    write_csv(OUT / "offline_model_metrics.csv", rows)
    write_csv(OUT / "offline_model_scores.csv", predictions)
    q_rows = list(csv.DictReader((OUT / "q_baseline_metrics.csv").open(newline="", encoding="utf-8")))
    aggregate = []
    names = ("success_pairwise_accuracy", "within_source_pairwise_accuracy_mean",
             "spearman_discounted_return", "within_source_spearman_mean",
             "top10_success", "top20_success", "top50_success")
    for split in ("test", "stress"):
        for method in METHODS:
            subset = [row for row in rows if row["test_split"] == split and row["model"] == method]
            for metric in names:
                mean, lo, hi = interval([row[metric] for row in subset])
                if metric.endswith("success") or "accuracy" in metric:
                    lo, hi = max(0.0, lo), min(1.0, hi)
                elif "spearman" in metric:
                    lo, hi = max(-1.0, lo), min(1.0, hi)
                aggregate.append({"test_split": split, "model": method, "metric": metric,
                                  "mean": mean, "ci95_low": lo, "ci95_high": hi,
                                  "n_model_seeds": len(subset)})
        for method, mode in (("E0", "best"), ("E2", "best"), ("E2", "final"),
                             ("B1", "best"), ("diag_B", "best"),
                             ("Q0", "best"), ("QGT", "best")):
            subset = [row for row in q_rows if row["test_split"] == split and
                      row["model"] == method and row["checkpoint_mode"] == mode]
            for metric in names:
                mean, lo, hi = interval([float(row[metric]) for row in subset])
                if metric.endswith("success") or "accuracy" in metric:
                    lo, hi = max(0.0, lo), min(1.0, hi)
                elif "spearman" in metric:
                    lo, hi = max(-1.0, lo), min(1.0, hi)
                aggregate.append({"test_split": split, "model": f"{method}_{mode}",
                                  "metric": metric, "mean": mean,
                                  "ci95_low": lo, "ci95_high": hi,
                                  "n_model_seeds": len(subset)})
    write_csv(OUT / "offline_model_aggregate.csv", aggregate)
    print("Scored four independent offline models and seven existing Q controls on both 1000-episode tests")
    for row in aggregate:
        if row["metric"] in ("success_pairwise_accuracy", "within_source_pairwise_accuracy_mean", "top20_success"):
            print(row["test_split"], row["model"], row["metric"],
                  f"{row['mean']:.3f} [{row['ci95_low']:.3f},{row['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
