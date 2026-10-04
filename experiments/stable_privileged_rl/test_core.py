"""CPU checks for rollback and trajectory metadata semantics."""

import tempfile
from pathlib import Path

import h5py
import torch

from auxiliary_learning.gt_prediction import AuxiliarySAC
from auxiliary_learning.multitask_encoder import MultiTaskSAC
from checkpoint_manager.rollback import RollbackMonitor
from replay.value_weighted_replay_v2 import ValueWeightedTrajectoryReplay
from evaluation.phase4_actor import RobotPolicy


def test_rollback():
    agent = AuxiliarySAC(26, 11, 7, device="cpu")
    monitor = RollbackMonitor(threshold=.25, patience=2)
    monitor.observe(agent, .75)
    original = {k:v.clone() for k,v in agent.actor.state_dict().items()}
    with torch.no_grad():
        next(agent.actor.parameters()).add_(1)
    assert not monitor.observe(agent, .25).rollback
    assert monitor.observe(agent, .25).rollback
    assert all(torch.equal(v, agent.actor.state_dict()[k]) for k,v in original.items())
    assert monitor.rollback_count == 1


def test_trajectory():
    agent = AuxiliarySAC(26, 11, 7, device="cpu")
    replay = ValueWeightedTrajectoryReplay(16,26,11,7,2,device="cpu")
    def step(rewards, dones):
        batch={"robot":torch.zeros(2,26),"privileged":torch.zeros(2,11),
               "action":torch.zeros(2,7),"reward":torch.tensor(rewards).reshape(2,1).float(),
               "next_robot":torch.zeros(2,26),"next_privileged":torch.zeros(2,11),
               "done":torch.tensor(dones).reshape(2,1).float()}
        replay.add_step(batch,torch.tensor(dones).bool(),torch.tensor([1.,0.]),agent)
    step([1,2],[False,False])
    step([3,4],[True,False])
    assert replay.complete_count == 1
    assert torch.allclose(replay.return_to_go[[0,2]],torch.tensor([1+3*.99,3.]))
    assert torch.isnan(replay.return_to_go[1])
    assert replay.success[0] == 1
    obj=replay.trajectory(0).as_dict()
    assert obj['state'].shape == (2,37) and 'quality_score' in obj
    assert replay.sample(4,mode="uniform")["action"].shape == (4,7)
    replay.refresh_weights()
    assert not replay.weighting_active
    replay.episode_summaries=[{'value':float(i),'return':float(i)} for i in range(64)]
    replay.refresh_weights()
    assert replay.weighting_active and replay.calibration_rho > .9
    assert torch.all((replay.quality_score[:replay.size][replay.episode_id[:replay.size]>=0] >= 0) &
                     (replay.quality_score[:replay.size][replay.episode_id[:replay.size]>=0] <= 1))
    assert replay.sample(4,mode='weighted')['action'].shape == (4,7)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'trajectory.h5'
        replay.export_hdf5(path)
        with h5py.File(path) as h5:
            assert len(h5['action']) == 4
            assert all(k in h5 for k in ('state','action','reward','next_state','return',
                                         'value','success','quality_score'))


def test_multitask_update():
    agent=MultiTaskSAC(26,11,7,device='cpu',aux_weight=.1)
    gt=torch.zeros(8,11)
    gt[:,9:11]=torch.randint(0,2,(8,2)).float()
    batch={'robot':torch.randn(8,26),'privileged':gt,'action':torch.randn(8,7).tanh(),
           'reward':torch.randn(8,1),'next_robot':torch.randn(8,26),
           'next_privileged':gt.clone(),'done':torch.zeros(8,1)}
    metrics=agent.update(batch,bc_batch=batch,bc_weight=10)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    agent.aux_weight=0
    assert agent.act(torch.zeros(1,26)).shape == (1,7)
    deploy=RobotPolicy()
    deploy.load_state_dict({k:v for k,v in agent.actor.state_dict().items()
                            if k.startswith('encoder.') or k.startswith('policy_head.')})
    probe=torch.randn(16,26)
    with torch.no_grad():
        assert torch.allclose(deploy(probe),agent.actor(probe,deterministic=True)[0],atol=1e-7)


if __name__ == '__main__':
    test_rollback()
    test_trajectory()
    test_multitask_update()
    print('Phase 4 rollback and trajectory replay CPU checks passed')
