"""Check deployable actor and critic fixture-information boundaries."""

import json
import torch

from algorithms.asymmetric_sac import AsymmetricSAC
from algorithms.door_phase3_sac import MaskedSAC
from auxiliary_learning.gt_prediction import AuxiliarySAC
from privileged_variants import gt_minimal, gt_medium, gt_full


def main():
    torch.set_num_threads(2)
    robot = torch.randn(8,26)
    gt = torch.randn(8,11)
    changed = gt + torch.randn_like(gt) * 10
    results = {}
    for label, indices in (('B1',gt_minimal.INDICES),('B2',gt_medium.INDICES),
                           ('B3',gt_full.INDICES)):
        agent = MaskedSAC(26,11,7,indices,device='cpu')
        a = agent.act(robot,gt,deterministic=True)
        b = agent.act(robot,changed,deterministic=True)
        assert torch.equal(a,b), f'{label} actor leaked fixture GT'
        expected=torch.cat((robot,gt[:,indices]),dim=-1)
        assert torch.equal(agent.critic_input(robot,gt),expected)
        results[label]={'actor_ignores_GT':True,'critic_GT_columns':list(indices)}
    base=AsymmetricSAC(26,11,7,'A',device='cpu')
    assert torch.equal(base.act(robot,gt,deterministic=True),
                       base.act(robot,changed,deterministic=True))
    torch.manual_seed(31337)
    e0=AuxiliarySAC(26,11,7,device='cpu',aux_weight=0)
    torch.manual_seed(31337)
    e1=AuxiliarySAC(26,11,7,device='cpu',aux_weight=.1)
    assert all(torch.equal(e0.actor.state_dict()[k],e1.actor.state_dict()[k])
               for k in e0.actor.state_dict())
    assert torch.equal(e0.act(robot,gt,deterministic=True),
                       e0.act(robot,changed,deterministic=True))
    assert torch.equal(e1.act(robot,gt,deterministic=True),
                       e1.act(robot,changed,deterministic=True))
    results['E0_E1']={'same_initial_actor_weights':True,'actors_ignore_GT':True}
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
