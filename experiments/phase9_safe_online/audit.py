"""Verify initialization, KL formula, sampling routes, replay sources and integrity."""
import hashlib
import json
from pathlib import Path
import torch
from auxiliary_learning.gt_prediction import EncodedGaussianActor
from safe_online.anchor_policy import AnchoredSAC
from safe_online.kl_constraint import normal_kl, anchor_kl, feasible_entropy_target
from safe_online.std_schedule import ControlledActor, StdSchedule
from safe_online.safe_update import SafeUpdate
from replay.success_buffer import SuccessReplay
from algorithms.replay import ReplayBuffer, FIELDS
from experiments.phase9_safe_online.protocol import ARMS, PROTOCOL
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase9_safe_online"


def main():
    immutable = json.loads((ROOT/"results"/"phase8_progress_rl"/"completion_manifest.json").read_text())["input_hashes_preserved"]
    for name, digest in immutable.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    hashes = {}
    reference = torch.randn(128, 26)
    for seed in range(5):
        path = ROOT/"checkpoints"/"phase8_progress_rl"/f"P8B_seed{seed}"/"best.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        old = EncodedGaussianActor(26, 7)
        old.load_state_dict(state["actor"])
        for arm in ARMS:
            agent = AnchoredSAC(state, "cpu", ARMS[arm]["kl"])
            assert torch.equal(agent.actor(reference, True)[0], old(reference, True)[0])
            assert all(torch.equal(v, agent.checkpoint()[k][name]) for k in ("critic", "target_critic") for name, v in state[k].items())
            torch.manual_seed(5151)
            old_action, old_logp = old(reference)
            torch.manual_seed(5151)
            action, logp = agent.actor(reference)
            assert torch.equal(action, old_action) and torch.equal(logp, old_logp)
            agent.actor.cap = agent.anchor.cap = .01
            assert agent.actor.distribution(reference)[1].exp().max() <= .010001
            assert abs(float(anchor_kl(agent.actor, agent.anchor, reference))) < 1e-6
            assert feasible_entropy_target(agent.anchor, reference) < -7
        hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    m, ref = torch.randn(128, 7), torch.randn(128, 7)
    log_std, ref_std = torch.randn(128, 7), torch.randn(128, 7)
    expected = torch.distributions.kl_divergence(torch.distributions.Normal(m, log_std.exp()),
        torch.distributions.Normal(ref, ref_std.exp())).sum(-1)
    assert torch.allclose(normal_kl(m, log_std, ref, ref_std), expected, rtol=1e-5, atol=1e-5)
    pools = []
    for tag in (1., 2., 3.):
        pool = ReplayBuffer(8, 26, 11, 7, "cpu")
        pool.add({key: torch.full((8, pool.data[key].shape[1]), tag) for key in FIELDS})
        pools.append(pool)
    replay_counts = {}
    for ratio in (1., .5, .3):
        sampler = SuccessReplay(pools[0], pools[2], pools[1], ratio)
        batch = sampler.sample()
        n_expert = round(256*ratio)
        assert (batch["robot"][:, 0] == 1).sum() == n_expert
        assert sum(sampler.draws.values()) == 256
        replay_counts[str(ratio)] = sampler.draws
    guard = SafeUpdate()
    metric = dict(success=1., max_angle=1.01, final_angle=1.01, progress=1., regression_event=0.)
    candidate = dict(metric, success=.75)
    assert guard.compare(dict(deterministic=metric, policy=metric), dict(deterministic=metric, policy=candidate)) == ["policy:success"]
    schedule = StdSchedule("anneal")
    assert schedule.cap(0)==.1 and abs(schedule.cap(100000)-.01)<1e-9
    result = dict(input_hashes_preserved=immutable, phase8_anchor_sha256=hashes,
        matched_full_checkpoint_initialization=True, original_sampling_bitwise_equal=True,
        analytic_kl_matches_torch=True, executed_anchor_kl_initially_zero=True,
        replay_source_counts=replay_counts, controlled_entropy_target_feasible=True,
        protocol=PROTOCOL)
    (OUT/"preflight_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
