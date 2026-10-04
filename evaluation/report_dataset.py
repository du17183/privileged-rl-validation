"""Audit a collected HDF5 expert dataset and write dataset_report.md."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/dataset_report.md"))
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as h5:
        episodes = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
        if not episodes:
            raise RuntimeError("No trajectories in HDF5")
        lengths = np.array([len(ep["action"]) for ep in episodes])
        required = ("observation", "state", "action", "reward", "next_observation", "next_state", "done")
        for ep in episodes:
            if any(key not in ep for key in required):
                raise RuntimeError(f"Missing required fields in {ep.name}")
            if any(len(ep[key]) != len(ep["action"]) for key in required):
                raise RuntimeError(f"Inconsistent transition lengths in {ep.name}")
            if not bool(ep.attrs.get("success", False)):
                raise RuntimeError(f"Non-successful expert trajectory: {ep.name}")
        attempts = int(h5.attrs.get("attempts", len(episodes)))
        interactions = h5.attrs.get("simulator_interactions")
        sources = json.loads(h5.attrs.get("sources", "[]"))
        state = np.concatenate([ep["state"][:] for ep in episodes])
        provenance = "\n".join(
            f"  - {item['trajectories']} trajectories, action noise std {item['arm_noise_std']}, "
            f"seed {item['seed']}, {item.get('attempts', '?')} attempts"
            for item in sources
        )
        source_detail = f"- Source runs:\n{provenance}" if sources else ""
        report = f"""# Drawer expert dataset

- File: `{args.dataset.resolve()}`
- Generation: Cartesian waypoint planner, differential IK, Panda gripper control
- Successful trajectories: {len(episodes)}
- Total attempted episodes: {attempts}
- Collection success rate: {len(episodes) / attempts:.3f}
- Mean / median / min / max episode length: {lengths.mean():.1f} / {np.median(lengths):.0f} / {lengths.min()} / {lengths.max()} steps
- Total saved transitions: {lengths.sum()}
- Total collection interactions including failed/partial episodes: {int(interactions) if interactions is not None else 'not recorded'}
- Robot observation dimension: {episodes[0]['observation'].shape[1]}
- Privileged state dimension: {episodes[0]['state'].shape[1]}
- Action dimension: {episodes[0]['action'].shape[1]}
- Drawer displacement range in saved states: {state[:, 0].min():.4f}–{state[:, 0].max():.4f} m
- Contact flag activation, left / right: {state[:, -2].mean():.3f} / {state[:, -1].mean():.3f}
{source_detail}

Each trajectory contains `observation`, `state`, `action`, `reward`, `next_observation`, `next_state`, `done`, `terminated`, and `truncated`. The `state` columns are drawer position, drawer velocity, handle position, handle quaternion (wxyz), left contact, and right contact. Only successful trajectories are saved; collection success rate uses all attempts.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
