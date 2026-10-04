"""Check exact feature overlap and outcome variation in locked Phase 5 splits."""

import csv
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def main():
    data = np.load(OUT / "episode_features.npz")
    rows = []
    for split in ("train", "validation", "test", "stress"):
        mask = data["split"] == split
        for source in np.unique(data["source"][mask]):
            group = mask & (data["source"] == source)
            rows.append({"split": split, "source": str(source),
                         "episodes": int(group.sum()), "successes": int(data["success"][group].sum()),
                         "mean_length": float(data["length"][group].mean()),
                         "min_length": int(data["length"][group].min()),
                         "ended_by_step50": int(np.sum(data["length"][group] <= 50)),
                         "ended_by_step100": int(np.sum(data["length"][group] <= 100))})
    feature_names = ("sequence", "early50", "early100", "start_robot")
    sets = {name: {} for name in feature_names}
    for name in feature_names:
        for split in ("train", "validation", "test", "stress"):
            mask = data["split"] == split
            sets[name][split] = {digest(array) for array in data[name][mask]}
    with (OUT / "split_overlap.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("feature", "split_a", "split_b", "identical_overlap"))
        writer.writeheader()
        for feature in feature_names:
            for a in ("train", "validation"):
                for b in ("test", "stress"):
                    overlap = len(sets[feature][a] & sets[feature][b])
                    writer.writerow({"feature": feature, "split_a": a, "split_b": b,
                                     "identical_overlap": overlap})
                    print(feature, a, b, "exact overlap", overlap)
    with (OUT / "source_outcome_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
