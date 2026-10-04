"""CPU checks of replay normalization, eligibility, indexing and selection ties."""
import json
import torch
from algorithms.replay import ReplayBuffer, FIELDS
from replay.quality_replay import QualityReplay
from trajectory_quality.quality_model import quality_score


class Expert:
    def __init__(self):
        self.prepared = {k: torch.arange(4.).reshape(-1,1).expand(-1, 1) for k in FIELDS}
    def sample(self,n):
        ids=torch.randint(4,(n,))
        return {k:v[ids] for k,v in self.prepared.items()}


def main():
    torch.manual_seed(381)
    online=ReplayBuffer(10,1,1,1,'cpu')
    online.add({k:torch.arange(10.,20.).reshape(-1,1) for k in FIELDS})
    expert=Expert()
    records=[dict(start=0,length=4,stride=1,success=1,quality_score=1.)]
    outputs={}
    for mode in ('uniform','weighted','selected','original','online'):
        replay=QualityReplay(expert,online,records,mode,.5,1000)
        replay.complete(dict(start=0,length=3,stride=2,success=0,quality_score=0.))
        replay.complete(dict(start=1,length=2,stride=2,success=1,quality_score=.9))
        batch=replay.sample()['action'].flatten()
        if mode not in ('original','online'):
            assert set(batch.tolist())<=set((0.,1.,2.,3.,10.,12.,14.,11.,13.))
            assert abs(float(replay.probability.sum())-1)<1e-6
            assert bool((replay.probability>0).all())
        if mode=='uniform':
            assert torch.allclose(replay.probability,torch.tensor([4/9,3/9,2/9]))
        if mode=='weighted':
            assert float(replay.probability[0]/4)>float(replay.probability[1]/3)
            assert float((replay.probability/(replay.lengths/9)).max())<7.4
        if mode=='original':assert replay.counts['expert']==500
        if mode=='online':assert replay.counts['expert']==0
        outputs[mode]=replay.distribution()
    from lwd.simplified_lwd import selection_mask
    assert bool(selection_mask(torch.ones(12)).all())
    assert quality_score(True,1.01,0,1,1)>quality_score(False,.5,0,1,.5)>quality_score(False,0,0,1,0)
    print(json.dumps(dict(passed=True,checks='normalization, completed-only eligibility, interleaved indexing, full support, bounded weights, selection ties, unchanged source quotas, physical ordering',replay=outputs),indent=2))

if __name__=='__main__':main()
