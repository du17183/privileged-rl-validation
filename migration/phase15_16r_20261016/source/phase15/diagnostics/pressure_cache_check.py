"""Regression: a finished pressure clone must never take over twice."""
import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from phase15.policy_client import ModularPolicy


def main():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'cohort.h5'
        with h5py.File(path, 'w') as h:
            for i in range(64):
                h.create_dataset(f'state_{i:04d}/prefix_actions', data=np.zeros((2, 7)))
        policy = ModularPolicy.__new__(ModularPolicy)
        torch.nn.Module.__init__(policy)
        policy.planner, policy.arm, policy.seed = None, 'A', 123
        policy.mean, policy.std = np.zeros(39), np.ones(39)
        calls = []
        def rpc(request):
            calls.append(request)
            return dict(actions=np.ones((len(request['raw']), 10, 7)), latency_s=0., service_s=0., clip_fraction=0.)
        policy.rpc = rpc
        policy.configure(path)
        raw = np.zeros((32, 39), np.float32)
        raw[:, 21] = raw[:, 35] = 1
        def step(ages):
            raw[:, 25] = np.asarray(ages) / 600
            policy.forward(torch.tensor(raw))
        def done(ids):
            mask = torch.zeros(32, dtype=torch.bool)
            mask[ids] = True
            policy.observe(torch.tensor(raw[:, :26]), torch.zeros((32, 11)), mask)
        step(np.zeros(32))
        step(np.full(32, 3))
        assert policy.handoff_replans == 32
        assert all(r['refreshed'] and r['cache_cleared'] for r in policy.handoff_records)
        done([0])
        ages = np.full(32, 4)
        ages[0] = 0
        step(ages)
        ages[0] = 3
        step(ages)
        assert not policy.active[0]
        assert policy.handoff_replans == 32
        done(list(range(1, 32)))
        step(np.zeros(32))
        step(np.full(32, 3))
        done(list(range(32)))
        assert policy.handoff_replans == 64
        assert len({r['cohort_id'] for r in policy.handoff_records}) == 64
        assert len(policy.telemetry) == 64
        print('PRESSURE CACHE REGRESSION PASSED: 64 unique physical takeovers; completed clone remained inactive')


if __name__ == '__main__':
    main()
