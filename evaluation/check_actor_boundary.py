"""Verify fixture GT cannot affect the A/B deployment actor interface."""

import json
from pathlib import Path

import torch

from algorithms.asymmetric_sac import AsymmetricSAC


def main():
    torch.manual_seed(17)
    robot = torch.randn(4, 26)
    fixture = torch.randn(4, 11)
    report = {}
    for variant in "AB":
        agent = AsymmetricSAC(26, 11, 7, variant, device="cpu")
        without_fixture = agent.act(robot, deterministic=True)
        with_fixture = agent.act(robot, fixture, deterministic=True)
        ignored = bool(torch.equal(without_fixture, with_fixture))
        if not ignored or agent.actor_dim != 26:
            raise RuntimeError(f"{variant} actor is affected by fixture GT")
        report[variant] = {"actor_input_dim": agent.actor_dim, "fixture_ignored": ignored}
    upper_bound = AsymmetricSAC(26, 11, 7, "C", device="cpu")
    try:
        upper_bound.act(robot, deterministic=True)
    except ValueError:
        fixture_required = True
    else:
        fixture_required = False
    if not fixture_required or upper_bound.actor_dim != 37:
        raise RuntimeError("C actor did not require fixture GT")
    report["C"] = {"actor_input_dim": upper_bound.actor_dim, "fixture_required": fixture_required}
    path = Path(__file__).resolve().parents[1] / "results/actor_gt_isolation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
