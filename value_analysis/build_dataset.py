"""Build leakage-controlled episode and transition arrays from frozen rollouts."""

import csv
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration"
POLICIES = (("E0", "best"), ("E0", "final"), ("E25", "final"),
            ("E100", "best"), ("E100", "final"), ("E100R1", "final"),
            ("Q0", "best"), ("Q0", "final"))
FIELDS = ("observation", "state", "action", "reward", "next_observation",
          "next_state", "done", "mc_return_to_go")


def main():
    episode = {key: [] for key in ("key", "source", "split", "seed", "sequence",
                                   "start_robot", "start_gt", "start_action", "success",
                                   "return_undiscounted", "return_discounted", "final_angle_rad",
                                   "length")}
    transition = {key: [] for key in ("episode_index", "robot", "gt", "action",
                                      "reward", "next_robot", "next_action", "done", "mc_return")}
    source_rows = []
    for seed in range(5):
        # Seed 4 is the prespecified 1,000-episode primary test. Seed 0 is
        # a second untouched, more mixed-outcome stress test. Neither can
        # influence model fitting or early stopping.
        split = {0: "stress", 1: "train", 2: "train", 3: "validation", 4: "test"}[seed]
        for variant, mode in POLICIES:
            source = f"{variant}_{mode}_seed{seed}"
            path = OUT / "frozen_rollouts" / f"{source}.h5"
            with h5py.File(path, "r") as h5:
                names = sorted(h5.keys())
                if len(names) != 125 or not bool(h5.attrs.get("complete", False)):
                    raise RuntimeError(f"Incomplete rollout source: {path}")
                successful = 0
                for name in names:
                    group = h5[name]
                    data = {field: group[field][:].astype(np.float32) for field in FIELDS}
                    length = len(data["action"])
                    if not 1 <= length <= 600 or data["observation"].shape[1] != 26 or data["state"].shape[1] != 11:
                        raise RuntimeError(f"Bad trajectory shape: {source}/{name}")
                    if not bool(data["done"][-1, 0]):
                        raise RuntimeError(f"Missing terminal step: {source}/{name}")
                    index = len(episode["key"])
                    sample = np.linspace(0, length - 1, 32).astype(np.int64)
                    sequence = np.concatenate((data["observation"][sample],
                                               data["action"][sample]), axis=1)
                    success = int(bool(group.attrs["success"]))
                    successful += success
                    for key, value in (("key", f"{source}/{name}"), ("source", f"{variant}_{mode}"),
                                       ("split", split), ("seed", seed), ("sequence", sequence),
                                       ("start_robot", data["observation"][0]),
                                       ("start_gt", data["state"][0]),
                                       ("start_action", data["action"][0]),
                                       ("success", success),
                                       ("return_undiscounted", float(group.attrs["episode_return"])),
                                       ("return_discounted", float(data["mc_return_to_go"][0, 0])),
                                       ("final_angle_rad", float(group.attrs["final_angle_rad"])),
                                       ("length", length)):
                        episode[key].append(value)
                    sample64 = np.linspace(0, length - 1, 64).astype(np.int64)
                    next_action = np.zeros((64, 7), dtype=np.float32)
                    not_last = sample64 < length - 1
                    next_action[not_last] = data["action"][sample64[not_last] + 1]
                    for key, value in (("episode_index", np.full(64, index, dtype=np.int32)),
                                       ("robot", data["observation"][sample64]),
                                       ("gt", data["state"][sample64]),
                                       ("action", data["action"][sample64]),
                                       ("reward", data["reward"][sample64]),
                                       ("next_robot", data["next_observation"][sample64]),
                                       ("next_action", next_action),
                                       ("done", data["done"][sample64]),
                                       ("mc_return", data["mc_return_to_go"][sample64])):
                        transition[key].append(value)
                source_rows.append({"source": f"{variant}_{mode}", "seed": seed, "split": split,
                                    "episodes": len(names), "successful": successful,
                                    "success_rate": successful / len(names)})
    np.savez_compressed(OUT / "episode_features.npz", **{
        key: np.asarray(values) for key, values in episode.items()})
    np.savez_compressed(OUT / "transition_features.npz", **{
        key: np.concatenate(values) for key, values in transition.items()})
    with (OUT / "source_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    for split in ("train", "validation", "test", "stress"):
        subset = [row for row in source_rows if row["split"] == split]
        print(split, "episodes", sum(row["episodes"] for row in subset),
              "successful", sum(row["successful"] for row in subset))
    print("Wrote episode_features.npz and transition_features.npz")


if __name__ == "__main__":
    main()
