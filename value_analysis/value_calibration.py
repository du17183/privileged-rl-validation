"""Offline value calibration; never modifies the policy or Door environment."""

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.asymmetric_sac import TwinQ, mlp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_calibration"
CKPT = ROOT / "checkpoints" / "value_calibration"


def e2_critic_state(seed):
    with (ROOT / "results" / "door_privileged_ablation" /
          f"eval_E2_seed{seed}.csv").open(newline="") as stream:
        row = max(csv.DictReader(stream), key=lambda item: float(item["success_rate"]))
    step = int(row["env_steps"])
    path = ROOT / "checkpoints" / "door_privileged_ablation" / f"E2_seed{seed}" / f"step_{step}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["critic"], step


def prepare(device):
    episode = np.load(OUT / "episode_features.npz")
    transition = np.load(OUT / "transition_features.npz")
    split = episode["split"][transition["episode_index"]]
    mask = {name: torch.as_tensor(split == name, device=device) for name in ("train", "validation")}
    arrays = {key: torch.as_tensor(transition[key], dtype=torch.float32, device=device)
              for key in ("robot", "action", "reward", "next_robot", "next_action", "done", "mc_return")}
    arrays["input"] = torch.cat((arrays["robot"], arrays["action"]), dim=1)
    return arrays, mask


@torch.no_grad()
def validation_loss(method, model, arrays, mask, mean, scale):
    x = arrays["input"][mask["validation"]]
    target = arrays["mc_return"][mask["validation"]]
    losses = []
    for start in range(0, len(x), 8192):
        if method == "MC":
            pred = model(x[start:start + 8192]) * scale + mean
        else:
            robot = x[start:start + 8192, :26]
            action = x[start:start + 8192, 26:]
            pred = torch.minimum(*model(robot, action))
        losses.append(F.mse_loss(pred, target[start:start + 8192], reduction="sum"))
    return float(torch.stack(losses).sum() / len(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("MC", "TD0", "TDMC"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=4000)
    args = parser.parse_args()
    torch.manual_seed(41000 + args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    arrays, mask = prepare(device)
    train_indices = torch.nonzero(mask["train"], as_tuple=False).flatten()
    target_train = arrays["mc_return"][mask["train"]]
    mean = float(target_train.mean()) if args.method == "MC" else 0.0
    scale = float(target_train.std().clamp_min(1e-4)) if args.method == "MC" else 1.0
    if args.method == "MC":
        model = mlp(33, 1).to(device)
        target_model = None
        source_step = -1
    else:
        model = TwinQ(26, 7).to(device)
        source, source_step = e2_critic_state(args.seed)
        model.load_state_dict(source)
        target_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_loss = float("inf")
    best_state = None
    no_improvement = 0
    history = []
    for update in range(args.updates + 1):
        if update % 250 == 0:
            val = validation_loss(args.method, model, arrays, mask, mean, scale)
            history.append({"method": args.method, "seed": args.seed,
                            "updates": update, "validation_mc_mse": val})
            if val < best_loss - 1e-5:
                best_loss = val
                best_state = copy.deepcopy(model.state_dict())
                best_update = update
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= 6:
                break
        if update == args.updates:
            break
        index = train_indices[torch.randint(len(train_indices), (1024,), device=device)]
        if args.method == "MC":
            target = (arrays["mc_return"][index] - mean) / scale
            prediction = model(arrays["input"][index])
            loss = F.mse_loss(prediction, target)
        else:
            robot, action = arrays["robot"][index], arrays["action"][index]
            q1, q2 = model(robot, action)
            with torch.no_grad():
                next_q = torch.minimum(*target_model(arrays["next_robot"][index],
                                                     arrays["next_action"][index]))
                td = arrays["reward"][index] + 0.99 * (1 - arrays["done"][index]) * next_q
            td_loss = F.mse_loss(q1, td) + F.mse_loss(q2, td)
            mc_loss = F.mse_loss(q1, arrays["mc_return"][index]) + \
                      F.mse_loss(q2, arrays["mc_return"][index])
            loss = td_loss + (1.0 if args.method == "TDMC" else 0.0) * mc_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if target_model is not None:
            with torch.no_grad():
                for target, current in zip(target_model.parameters(), model.parameters()):
                    target.lerp_(current, 0.005)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"method": args.method, "seed": args.seed, "model": best_state,
                "mc_mean": mean, "mc_scale": scale, "source_e2_steps": source_step,
                "best_update": best_update, "validation_mc_mse": best_loss},
               CKPT / f"{args.method}_seed{args.seed}.pt")
    with (OUT / f"train_{args.method}_seed{args.seed}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"{args.method} seed={args.seed} best_update={best_update} "
          f"validation_mc_mse={best_loss:.5f}", flush=True)


if __name__ == "__main__":
    main()
