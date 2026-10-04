"""Read-only integrity check for the completed Panda Door Phase 5 package."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"
PILOT = OUT / "online_pilot"
ARMS = ("uniform", "quality", "quality_return", "quality_offline")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    frozen = sorted((OUT / "frozen_rollouts").glob("*.h5"))
    if len(frozen) != 40:
        raise RuntimeError(f"Expected 40 frozen HDF5 files, found {len(frozen)}")
    episodes = 0
    for path in frozen:
        with h5py.File(path, "r") as h5:
            if not bool(h5.attrs["complete"]) or int(h5.attrs["policy_updates"]) != 0 or len(h5) != 125:
                raise RuntimeError(f"Invalid frozen-policy HDF5: {path}")
            episodes += len(h5)
    data = np.load(OUT / "episode_features.npz")
    split_counts = {name: int(np.sum(data["split"] == name))
                    for name in ("train", "validation", "test", "stress")}
    if split_counts != {"train": 2000, "validation": 1000, "test": 1000, "stress": 1000}:
        raise RuntimeError(f"Split counts changed: {split_counts}")
    if episodes != len(data["key"]) or episodes != 5000:
        raise RuntimeError("Frozen HDF5 count and dataset arrays disagree")
    if len(csv_rows(OUT / "q_baseline_metrics.csv")) != 140:
        raise RuntimeError("Q baseline ranking audit incomplete")
    if len(csv_rows(OUT / "offline_model_metrics.csv")) != 64:
        raise RuntimeError("Offline model ranking audit incomplete")
    overlap = csv_rows(OUT / "split_overlap.csv")
    if any(int(row["identical_overlap"]) for row in overlap if row["feature"] == "early100"):
        raise RuntimeError("Fixed first-100-step input crosses split boundary")
    gate = json.loads((PILOT / "offline_gate_passed.json").read_text(encoding="utf-8"))
    protocol = json.loads((PILOT / "protocol_verification.json").read_text(encoding="utf-8"))
    if not gate["passed"] or not protocol["passed"] or len(protocol["rows"]) != 20:
        raise RuntimeError("Offline gate or online paired protocol failed")
    pilot_runs, heldout_checks = 0, 0
    for arm in ARMS:
        for seed in range(5):
            run = f"{arm}_seed{seed}"
            curve = csv_rows(PILOT / f"eval_{run}.csv")
            checks = csv_rows(PILOT / f"heldout_{run}.csv")
            if len(curve) != 11 or int(curve[-1]["env_steps"]) != 100000:
                raise RuntimeError(f"Incomplete online training: {run}")
            if len(checks) != 2 or any(int(row["heldout_episodes"]) != 64 for row in checks):
                raise RuntimeError(f"Incomplete independent actor test: {run}")
            pilot_runs += 1
            heldout_checks += len(checks)
    for name, expected in (("status.csv", 10), ("return_gate_status.csv", 5),
                           ("offline_gate_status.csv", 5), ("heldout_status.csv", 20)):
        rows = csv_rows(PILOT / name)
        if len(rows) != expected or any(int(row["exit_code"]) for row in rows):
            raise RuntimeError(f"Failed or missing job in {name}")
    for name in ("value_diagnosis.md", "value_quality_report.md"):
        if not (ROOT / "docs" / name).exists():
            raise RuntimeError(f"Missing report {name}")
    for name in ("success_vs_steps.png", "reward_vs_steps.png", "weighted_activation.png"):
        if not (PILOT / name).is_file():
            raise RuntimeError(f"Missing figure {name}")
    result = {"passed": True, "frozen_hdf5_files": len(frozen),
              "frozen_episodes": episodes, "split_counts": split_counts,
              "q_checkpoint_test_rows": 140, "offline_model_test_rows": 64,
              "pilot_runs": pilot_runs, "pilot_train_interactions": pilot_runs * 100000,
              "independent_heldout_checks": heldout_checks,
              "paired_initialization_verified": True}
    (OUT / "phase5_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
