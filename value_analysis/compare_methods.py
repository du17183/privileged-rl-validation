"""Five-seed paired offline ranking contrasts with exact sign-flip tests."""

import csv
from pathlib import Path

import numpy as np

from experiments.value_calibration.analyze_online_pilot import exact_signflip
from trajectory_quality.ranking_eval import interval, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"


def main():
    with (OUT / "offline_model_metrics.csv").open(newline="", encoding="utf-8") as stream:
        model = list(csv.DictReader(stream))
    with (OUT / "q_baseline_metrics.csv").open(newline="", encoding="utf-8") as stream:
        critic = list(csv.DictReader(stream))
    metrics = ("success_pairwise_accuracy", "within_source_pairwise_accuracy_mean",
               "spearman_discounted_return", "top20_success")
    result = []
    for split in ("test", "stress"):
        early = {int(row["model_seed"]): row for row in model if row["test_split"] == split
                 and row["model"] == "EARLY100"}
        controls = (("MC", model, "model_seed", None),
                    ("TD0", model, "model_seed", None),
                    ("E2_best_Q", critic, "critic_seed", "E2"),
                    ("Q0_best_Q", critic, "critic_seed", "Q0"))
        for label, source, seed_field, q_variant in controls:
            if q_variant is None:
                baseline = {int(row[seed_field]): row for row in source
                            if row["test_split"] == split and row["model"] == label}
            else:
                baseline = {int(row[seed_field]): row for row in source
                            if row["test_split"] == split and row["model"] == q_variant
                            and row["checkpoint_mode"] == "best"}
            if set(early) != set(baseline) or len(early) != 5:
                raise RuntimeError("Missing paired model seeds")
            for metric in metrics:
                difference = np.asarray([float(early[seed][metric]) -
                                         float(baseline[seed][metric]) for seed in range(5)])
                mean, lo, hi = interval(difference)
                result.append({"test_split": split, "comparison": f"EARLY100_minus_{label}",
                               "metric": metric, "mean_difference": mean,
                               "ci95_low": lo, "ci95_high": hi,
                               "exact_signflip_p": exact_signflip(difference)})
    write_csv(OUT / "offline_paired_comparisons.csv", result)
    for row in result:
        if row["metric"] == "success_pairwise_accuracy":
            print(row)


if __name__ == "__main__":
    main()
