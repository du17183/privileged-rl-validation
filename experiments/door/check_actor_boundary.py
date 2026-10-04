"""Verify fixture GT cannot change A/B/D actor output at inference."""
import json
from pathlib import Path
import torch
from algorithms.asymmetric_sac import AsymmetricSAC

torch.manual_seed(17)
robot = torch.randn(16, 26)
gt1 = torch.randn(16, 11)
gt2 = torch.randn(16, 11)
out = {}
for variant in ("A", "B", "C"):
    agent = AsymmetricSAC(26, 11, 7, variant, device="cpu")
    if variant == "C":
        try:
            agent.act(robot, deterministic=True)
            raise RuntimeError("C actor did not require GT")
        except ValueError:
            pass
        out[variant] = {"requires_gt": True,
                        "action_changes_with_gt": bool((agent.act(robot, gt1, True) != agent.act(robot, gt2, True)).any())}
    else:
        a = agent.act(robot, deterministic=True)
        b = agent.act(robot, gt1, deterministic=True)
        c = agent.act(robot, gt2, deterministic=True)
        out[variant] = {"requires_gt": False, "max_action_difference_with_gt":
                        float(torch.max(torch.abs(a-b)).item() + torch.max(torch.abs(a-c)).item())}
        if out[variant]["max_action_difference_with_gt"] != 0:
            raise RuntimeError(f"{variant} actor leaked GT")
out["D"] = dict(out["B"])
path = Path(__file__).resolve().parents[2] / "results/door/actor_gt_isolation.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
