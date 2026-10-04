"""Rank heldout frozen-policy Door trajectories using existing SAC Q models."""

import csv
from pathlib import Path

import numpy as np
import torch

from algorithms.asymmetric_sac import TwinQ
from auxiliary_learning.gt_prediction import EncodedGaussianActor
from value_analysis.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"
MODELS = (("E0", "phase3", ()), ("E2", "phase3", ()),
          ("B1", "phase3", (0,)), ("B2", "phase3", (0, 1)),
          ("diag_B", "phase3", tuple(range(11))),
          ("Q0", "phase4", ()), ("QGT", "phase4", ()))


def checkpoint(variant, phase, seed, mode):
    folder = "door_privileged_ablation" if phase == "phase3" else "stable_privileged_rl"
    with (ROOT / "results" / folder / f"eval_{variant}_seed{seed}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = max(rows, key=lambda item: float(item["success_rate"])) if mode == "best" else rows[-1]
    step = int(row["env_steps"])
    path = ROOT / "checkpoints" / folder / f"{variant}_seed{seed}" / f"step_{step}.pt"
    return torch.load(path, map_location="cpu", weights_only=False), step


@torch.no_grad()
def score(variant, phase, indices, state, robot, gt, action):
    if phase == "phase4":
        actor = EncodedGaussianActor(26, 7)
        actor.load_state_dict(state["actor"])
        features = actor.encoder(torch.from_numpy(robot))
        critic = TwinQ(256, 7)
    else:
        features = torch.from_numpy(np.concatenate((robot, gt[:, indices]), axis=1)
                                    if indices else robot)
        critic = TwinQ(features.shape[1], 7)
    critic.load_state_dict(state["critic"])
    critic.eval()
    q1, q2 = critic(features, torch.from_numpy(action))
    return torch.minimum(q1, q2).flatten().numpy()


def main():
    torch.set_num_threads(4)
    features = np.load(OUT / "episode_features.npz")
    rows, predictions = [], []
    for test_split in ("test", "stress"):
        test = features["split"] == test_split
        robot = features["start_robot"][test].astype(np.float32)
        gt = features["start_gt"][test].astype(np.float32)
        action = features["start_action"][test].astype(np.float32)
        truth = {key: features[key][test] for key in ("return_discounted", "return_undiscounted",
                                                      "success", "source", "key")}
        if len(robot) != 1000 or not (0 < truth["success"].sum() < len(robot)):
            raise RuntimeError(f"{test_split} must contain exactly 1000 mixed-outcome episodes")
        for variant, phase, indices in MODELS:
            for mode in ("best", "final"):
                for seed in range(5):
                    state, step = checkpoint(variant, phase, seed, mode)
                    values = score(variant, phase, indices, state, robot, gt, action)
                    result = evaluate(values, truth["return_discounted"],
                                      truth["return_undiscounted"], truth["success"], truth["source"])
                    rows.append({"model": variant, "phase": phase, "checkpoint_mode": mode,
                                 "critic_seed": seed, "checkpoint_steps": step,
                                 "test_split": test_split, **result})
                    predictions.extend({"model": variant, "phase": phase, "checkpoint_mode": mode,
                                        "critic_seed": seed, "test_split": test_split,
                                        "trajectory": key, "score": float(value)}
                                       for key, value in zip(truth["key"], values))
    for name, data in (("q_baseline_metrics.csv", rows), ("q_baseline_scores.csv", predictions)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"Evaluated {len(rows)} frozen Q checkpoint/split pairs on two disjoint 1000-episode tests")


if __name__ == "__main__":
    main()
