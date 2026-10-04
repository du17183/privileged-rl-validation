"""Fixed expert-state Q probe across unchanged Phase 2 Door checkpoints.

This distinguishes drift on an identical input distribution from drift caused
by changing online replay composition. It is a diagnostic, not a causal test.
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch

from algorithms.asymmetric_sac import AsymmetricSAC


ROOT = Path(__file__).resolve().parents[1]


def fixed_probe(path, per_trajectory=16, trajectories=16):
    fields = {"robot": [], "privileged": [], "action": [], "reward": [],
              "next_robot": [], "next_privileged": [], "done": []}
    datasets = {"robot": "observation", "privileged": "state", "action": "action",
                "reward": "reward", "next_robot": "next_observation",
                "next_privileged": "next_state", "done": "done"}
    with h5py.File(path, "r") as h5:
        names = sorted(name for name in h5 if name.startswith("traj_"))
        selected = np.linspace(0, len(names) - 1, trajectories, dtype=int)
        for j in selected:
            group = h5[names[j]]
            idx = np.linspace(0, len(group["action"]) - 1, per_trajectory, dtype=int)
            for key, dataset in datasets.items():
                array = np.asarray(group[dataset][...], dtype=np.float32)[idx]
                fields[key].append(array.reshape(per_trajectory, -1))
    return {key: torch.from_numpy(np.concatenate(value)) for key, value in fields.items()}


def checkpoint_probe(agent, batch):
    agent.actor.eval()
    agent.critic.eval()
    agent.target_critic.eval()
    robot, gt, action = batch["robot"], batch["privileged"], batch["action"]
    critic_obs = agent.critic_input(robot, gt)
    with torch.no_grad():
        q1_expert, q2_expert = agent.critic(critic_obs, action)
        expert_q = torch.minimum(q1_expert, q2_expert)
        policy_action = agent.act(robot, deterministic=True)
        q1_policy, q2_policy = agent.critic(critic_obs, policy_action)
        policy_q = torch.minimum(q1_policy, q2_policy)
        torch.manual_seed(20260929)
        next_action, next_logp = agent.actor(batch["next_robot"])
        tq1, tq2 = agent.target_critic(
            agent.critic_input(batch["next_robot"], batch["next_privileged"]), next_action)
        target = batch["reward"] + agent.cfg.gamma * (1.0 - batch["done"]) * (
            torch.minimum(tq1, tq2) - agent.log_alpha.exp() * next_logp)
        td_abs = (expert_q - target).abs().mean()
    action_for_grad = policy_action.detach().requires_grad_(True)
    q1_grad, q2_grad = agent.critic(critic_obs, action_for_grad)
    grad = torch.autograd.grad(torch.minimum(q1_grad, q2_grad).sum(), action_for_grad)[0]
    toward_expert = action - policy_action
    valid = toward_expert.norm(dim=-1) > 0.1
    alignment = torch.nn.functional.cosine_similarity(grad[valid], toward_expert[valid], dim=-1)
    return {"q_expert_mean": float(expert_q.mean()),
            "q_expert_variance": float(expert_q.var(unbiased=False)),
            "q_policy_mean": float(policy_q.mean()),
            "q_policy_variance": float(policy_q.var(unbiased=False)),
            "q_policy_minus_expert": float((policy_q - expert_q).mean()),
            "q_twin_disagreement": float((q1_expert - q2_expert).abs().mean()),
            "expert_td_abs": float(td_abs),
            "policy_action_q_grad_l2": float(grad.norm(dim=-1).mean()),
            "q_gradient_toward_expert_cos": float(alignment.mean()) if valid.any() else float('nan'),
            "q_gradient_toward_expert_fraction": float((alignment > 0).float().mean()) if valid.any() else float('nan'),
            "expert_action_gap_l2": float(toward_expert.norm(dim=-1).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "door")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "door_privileged_ablation" / "fixed_probe_phase2.csv")
    args = parser.parse_args()
    torch.set_num_threads(4)
    batch = fixed_probe(ROOT / "door_dataset" / "door_expert_1000.h5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for variant in ("A", "B"):
            for seed in range(5):
                paths = sorted((args.checkpoint_dir / f"{variant}_seed{seed}").glob("step_*.pt"),
                               key=lambda p: int(p.stem.split("_")[-1]))
                if len(paths) != 21:
                    raise RuntimeError(f"Expected 21 checkpoints: {variant} seed {seed}; got {len(paths)}")
                agent = AsymmetricSAC(26, 11, 7, variant, device="cpu")
                for path in paths:
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    agent.actor.load_state_dict(state["actor"])
                    agent.critic.load_state_dict(state["critic"])
                    agent.target_critic.load_state_dict(state["target_critic"])
                    agent.log_alpha.data.copy_(state["log_alpha"])
                    row = {"variant": variant, "seed": seed, "env_steps": state["env_steps"],
                           **checkpoint_probe(agent, batch)}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                print(f"probed {variant} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
