import torch

from asymmetric_sac import AsymmetricSAC


def test_variants():
    torch.manual_seed(17)
    n, robot_dim, gt_dim, action_dim = 32, 13, 5, 7
    robot = torch.randn(n, robot_dim)
    gt = torch.randn(n, gt_dim)
    action = torch.tanh(torch.randn(n, action_dim))
    batch = {
        "robot": robot,
        "privileged": gt,
        "action": action,
        "reward": torch.randn(n, 1),
        "next_robot": torch.randn(n, robot_dim),
        "next_privileged": torch.randn(n, gt_dim),
        "done": torch.zeros(n, 1),
    }
    for variant in ("A", "B", "C"):
        agent = AsymmetricSAC(robot_dim, gt_dim, action_dim, variant, device="cpu")
        before = agent.act(robot, gt, deterministic=True)
        if variant in ("A", "B"):
            assert torch.equal(before, agent.act(robot, gt * 100, deterministic=True))
        else:
            assert not torch.equal(before, agent.act(robot, gt * 100, deterministic=True))
        assert 0 <= agent.bc_step(robot, gt, action) < 4
        metrics = agent.update(batch)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        assert agent.act(robot, gt).shape == action.shape


if __name__ == "__main__":
    test_variants()
    print("A/B/C input routing, BC, and SAC update: PASS")
