"""Merge successful HDF5 planner trajectories without changing transition data."""

import argparse
import json
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    count = 0
    attempts = 0
    interactions = 0
    shapes = None
    provenance = []
    with h5py.File(args.output, "w") as target:
        for source_path in args.sources:
            with h5py.File(source_path, "r") as source:
                groups = [source[key] for key in sorted(source) if key.startswith("traj_")]
                if len(groups) != int(source.attrs["successful_episodes"]):
                    raise ValueError(f"Incomplete source: {source_path}")
                source_interactions = int(source.attrs.get("simulator_interactions", 0))
                provenance.append({
                    "path": str(source_path.resolve()),
                    "trajectories": len(groups),
                    "arm_noise_std": float(source.attrs.get("arm_noise_std", 0.0)),
                    "seed": int(source.attrs["seed"]),
                    "attempts": int(source.attrs.get("attempts", len(groups))),
                    "simulator_interactions": source_interactions,
                })
                attempts += int(source.attrs.get("attempts", len(groups)))
                interactions += source_interactions
                for group in groups:
                    if not bool(group.attrs["success"]):
                        raise ValueError(f"Failed trajectory: {group.name}")
                    shape = tuple(group[key].shape[1:] for key in ("observation", "state", "action"))
                    if shapes is None:
                        shapes = shape
                    elif shape != shapes:
                        raise ValueError(f"Incompatible observation/state/action shape: {group.name}")
                    name = f"traj_{count:05d}"
                    source.copy(group, target, name=name)
                    target[name].attrs["source_file"] = str(source_path)
                    count += 1
        if not 500 <= count <= 1000:
            raise ValueError(f"Expected 500-1000 successful trajectories, found {count}")
        target.attrs["successful_episodes"] = count
        target.attrs["attempts"] = attempts
        target.attrs["simulator_interactions"] = interactions
        target.attrs["sources"] = json.dumps(provenance)
        target.attrs["env_id"] = "Isaac-Open-Drawer-Franka-IK-Rel-v0"
        target.flush()
    print(json.dumps({"output": str(args.output), "successful_episodes": count, "sources": provenance}, indent=2))


if __name__ == "__main__":
    main()
