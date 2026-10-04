"""Check why the first paired pilot did or did not enable quality weighting."""

import csv
from pathlib import Path

import h5py
import numpy as np

from value_analysis.metrics import safe_corr, success_ranking_accuracy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"


def main():
    rows = []
    for arm in ("uniform", "quality", "quality_return", "quality_offline"):
        for seed in range(5):
            path = OUT / f"trajectories_{arm}_seed{seed}.h5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as h5:
                episode_id = h5["episode_id"][:]
                scores = h5["quality_score"][:]
                rewards = h5["reward"][:].reshape(-1)
            observed = []
            for index in np.unique(episode_id[episode_id >= 0]):
                mask = episode_id == index
                r = rewards[mask]
                observed.append((float(scores[mask][0]),
                                 float(np.dot(r, .99 ** np.arange(len(r)))),
                                 int(np.max(r) >= 600)))
            array = np.asarray(observed)
            finite = np.isfinite(array[:, 0])
            rows.append({"arm": arm, "seed": seed, "complete_episodes": len(observed),
                         "successful_online_episodes": int(array[:, 2].sum()),
                         "score_mean": float(np.mean(array[finite, 0])) if finite.any() else float("nan"),
                         "score_std": float(np.std(array[finite, 0])) if finite.any() else float("nan"),
                         "score_return_spearman": safe_corr(array[finite, 0], array[finite, 1],
                                                            "spearman") if finite.any() else float("nan"),
                         "score_success_auc": success_ranking_accuracy(array[finite, 0],
                                                                       array[finite, 2]) if finite.any() else float("nan")})
    if not rows:
        raise RuntimeError("No completed online pilot HDF5 files")
    with (OUT / "online_distribution_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
