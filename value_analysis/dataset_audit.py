"""Read-only audit of complete Phase 4 Door episodes available for Phase 5."""

import csv
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"


def inspect(path):
    with h5py.File(path, "r") as h5:
        episode_id = h5["episode_id"][:]
        valid = episode_id >= 0
        ids, first = np.unique(episode_id[valid], return_index=True)
        positions = np.flatnonzero(valid)[first]
        success = h5["success"][:][positions]
        returns = h5["return"][:][positions]
        length = np.bincount(episode_id[valid].astype(np.int64))
        return {
            "run": path.stem.removeprefix("trajectories_"),
            "episodes": int(len(ids)),
            "successful": int(np.sum(success > 0.5)),
            "success_fraction": float(np.mean(success > 0.5)),
            "return_min": float(np.min(returns)),
            "return_median": float(np.median(returns)),
            "return_max": float(np.max(returns)),
            "episode_length_min": int(np.min(length)),
            "episode_length_max": int(np.max(length)),
        }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "results" / "stable_privileged_rl" /
             f"trajectories_{variant}_seed{seed}.h5"
             for variant in ("Q0", "QGT", "QV") for seed in range(5)]
    rows = [inspect(path) for path in files]
    path = OUT / "dataset_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
