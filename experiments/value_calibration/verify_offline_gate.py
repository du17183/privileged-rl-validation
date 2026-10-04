"""Record the pre-online locked gate for the fixed first-100-step predictor."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration"


def main():
    with (OUT / "offline_model_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (OUT / "split_overlap.csv").open(newline="", encoding="utf-8") as stream:
        overlap = list(csv.DictReader(stream))
    checks = {}
    for split in ("test", "stress"):
        subset = [row for row in rows if row["test_split"] == split and
                  row["model"] == "EARLY100"]
        if len(subset) != 5:
            raise RuntimeError(f"Missing five EARLY100 model seeds on {split}")
        metric = lambda name: float(np.mean([float(row[name]) for row in subset]))
        checks[split] = {"success_pairwise_auc": metric("success_pairwise_accuracy"),
                         "within_source_auc": metric("within_source_pairwise_accuracy_mean"),
                         "top20_success": metric("top20_success"),
                         "uniform_success": metric("success_prevalence")}
    exact_overlap = sum(int(row["identical_overlap"]) for row in overlap
                        if row["feature"] == "early100")
    passed = (exact_overlap == 0 and all(
        value["success_pairwise_auc"] >= 0.8 and
        value["within_source_auc"] >= 0.65 and
        value["top20_success"] >= value["uniform_success"] + 0.10
        for value in checks.values()))
    artifact = {"passed": passed, "predictor": "EARLY100", "test_episodes": 1000,
                "stress_episodes": 1000, "training_episodes": 2000,
                "validation_episodes": 1000, "model_seeds": 5,
                "exact_early100_split_overlap": exact_overlap,
                "criteria": {"pairwise_auc": 0.8, "within_source_auc": 0.65,
                             "top20_above_uniform": 0.10},
                "observed": checks,
                "scope": "Frozen-policy offline ranking only; no policy gain implied."}
    path = OUT / "online_pilot" / "offline_gate_passed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise SystemExit("Offline gate failed: online pilot must not run")


if __name__ == "__main__":
    main()
