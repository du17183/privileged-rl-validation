"""Read-only expert HDF5 audit and replay loader for Phase 6.

BC is the offline policy baseline: Phase 6 B1/B2/B3 train the same BC actor
for 3,000 updates, save it as step_0, and evaluate it before any online step.
This module does not alter the expert dataset or run online policy updates.
"""

from pathlib import Path

import h5py

from algorithms.replay import ReplayBuffer


REQUIRED = ("observation", "state", "action", "reward", "next_observation",
            "next_state", "done")


def audit_demonstrations(path):
    path = Path(path)
    with h5py.File(path, "r") as h5:
        groups = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
        if len(groups) != 1000:
            raise ValueError(f"Expected 1,000 expert trajectories; found {len(groups)}")
        if any(not all(key in group for key in REQUIRED) for group in groups):
            raise ValueError("Expert trajectories have missing fields")
        successes = sum(bool(group.attrs["success"]) for group in groups)
        transitions = sum(len(group["action"]) for group in groups)
        shapes = {key: groups[0][key].shape[1:] for key in REQUIRED}
        return {"path": str(path.resolve()), "trajectories": len(groups),
                "successes": successes, "transitions": transitions,
                "field_shapes": {key: list(value) for key, value in shapes.items()}}


def load_demonstrations(path, device):
    with h5py.File(path, "r") as h5:
        groups = [h5[key] for key in sorted(h5) if key.startswith("traj_")]
        if len(groups) != 1000:
            raise ValueError(f"Expected 1,000 expert trajectories; found {len(groups)}")
        if any(not bool(group.attrs["success"]) or
               not all(key in group for key in REQUIRED) for group in groups):
            raise ValueError("Expected successful, complete expert trajectories")
        total = sum(len(group["action"]) for group in groups)
        first = groups[0]
        buffer = ReplayBuffer(total, first["observation"].shape[1],
                              first["state"].shape[1], first["action"].shape[1],
                              device=device)
        for group in groups:
            buffer.add({
                "robot": group["observation"][:],
                "privileged": group["state"][:],
                "action": group["action"][:],
                "reward": group["reward"][:],
                "next_robot": group["next_observation"][:],
                "next_privileged": group["next_state"][:],
                "done": group["done"][:],
            })
    return buffer


def fit_bc_holdout(path, output, seeds=range(5), updates=3000, device="cpu"):
    """Fit BC offline on 800 expert trajectories; score 200 disjoint ones."""
    import csv
    import json
    import numpy as np
    import torch
    from torch.nn import functional as F
    from auxiliary_learning.gt_prediction import EncodedGaussianActor

    audit = audit_demonstrations(path)
    with h5py.File(path, "r") as h5:
        names = sorted(key for key in h5 if key.startswith("traj_"))
        indices = np.random.default_rng(12345).permutation(len(names))
        train_names = [names[i] for i in indices[:800]]
        test_names = [names[i] for i in indices[800:]]
        def load_fields(keys):
            obs = np.concatenate([h5[key]["observation"][:] for key in keys])
            act = np.concatenate([h5[key]["action"][:] for key in keys])
            return (torch.as_tensor(obs, dtype=torch.float32, device=device),
                    torch.as_tensor(act, dtype=torch.float32, device=device))
        train_obs, train_act = load_fields(train_names)
        test_obs, test_act = load_fields(test_names)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        actor = EncodedGaussianActor(train_obs.shape[-1], train_act.shape[-1]).to(device)
        with torch.no_grad():
            random_mse = float(F.mse_loss(actor(test_obs, deterministic=True)[0], test_act))
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        torch.manual_seed(int(seed) + 100000)
        for _ in range(updates):
            idx = torch.randint(len(train_obs), (256,), device=device)
            prediction, _ = actor(train_obs[idx], deterministic=True)
            loss = F.mse_loss(prediction, train_act[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_idx = torch.arange(min(len(train_obs), 16384), device=device)
            train_mse = float(F.mse_loss(actor(train_obs[train_idx], deterministic=True)[0],
                                         train_act[train_idx]))
            test_mse = float(F.mse_loss(actor(test_obs, deterministic=True)[0], test_act))
        records.append((seed, random_mse, train_mse, test_mse))
        print(f"BC seed={seed}: random MSE={random_mse:.5f} train={train_mse:.5f} heldout={test_mse:.5f}", flush=True)
    with (output / "offline_bc_holdout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "random_action_mse", "bc_train_mse", "bc_heldout_mse"))
        writer.writerows(records)
    (output / "expert_dataset_audit.json").write_text(json.dumps({
        **audit, "train_trajectories": 800, "test_trajectories": 200,
        "partition_seed": 12345, "bc_updates": updates,
        "note": "Action regression is an offline diagnostic, not environment success."
    }, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="door_dataset/door_expert_1000.h5")
    parser.add_argument("--output", default="results/phase6_stable_online_rl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=3000)
    options = parser.parse_args()
    fit_bc_holdout(options.dataset, options.output, updates=options.updates,
                   device=options.device)
