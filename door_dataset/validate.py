"""Audit completed expert HDF5 trajectories and emit machine-readable stats."""

import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "door_dataset/door_expert_1000.h5"
TARGET = ROOT / "results/door/dataset_summary.json"
REQUIRED = ("observation", "robot_state", "state", "privileged_state", "action",
            "reward", "next_observation", "next_state", "done", "terminated", "truncated")

with h5py.File(SOURCE, "r") as h5:
    groups = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
    lengths = []
    hashes = set()
    contacts = []
    final_angles = []
    for group in groups:
        if not all(key in group for key in REQUIRED) or not bool(group.attrs["success"]):
            raise RuntimeError(f"Incomplete or failed trajectory: {group.name}")
        n = len(group["action"])
        if any(len(group[key]) != n for key in REQUIRED):
            raise RuntimeError(f"Inconsistent length: {group.name}")
        if n == 0 or group["done"][:-1].any() or not group["done"][-1, 0]:
            raise RuntimeError(f"Invalid terminal marker: {group.name}")
        if not group["terminated"][-1, 0] or group["truncated"][-1, 0]:
            raise RuntimeError(f"Non-successful terminal state: {group.name}")
        action = group["action"][:]
        if not np.isfinite(action).all() or not np.isfinite(group["state"][:]).all():
            raise RuntimeError(f"Nonfinite data: {group.name}")
        angle = float(group["next_state"][-1, 0])
        if angle <= 1.0:
            raise RuntimeError(f"Invalid final angle: {group.name} {angle}")
        lengths.append(n)
        hashes.add(hashlib.sha256(action.tobytes()).hexdigest())
        contacts.append(float(np.any(np.all(group["state"][:, -2:] > 0.5, axis=1))))
        final_angles.append(angle)
    summary = {
        "trajectory_count": len(groups), "attempts": int(h5.attrs["attempts"]),
        "success_fraction": len(groups) / int(h5.attrs["attempts"]),
        "unique_action_trajectories": len(hashes),
        "total_transitions": int(sum(lengths)),
        "mean_length": float(np.mean(lengths)), "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "robot_observation_dim": int(groups[0]["observation"].shape[1]),
        "privileged_state_dim": int(groups[0]["state"].shape[1]),
        "action_dim": int(groups[0]["action"].shape[1]),
        "ever_both_fingers_contact_fraction": float(np.mean(contacts)),
        "mean_final_angle_rad": float(np.mean(final_angles)),
        "simulator_interactions": int(h5.attrs["simulator_interactions"]),
    }
if summary["trajectory_count"] != 1000 or summary["unique_action_trajectories"] != 1000:
    raise RuntimeError("Expected 1000 distinct successful expert trajectories")
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
