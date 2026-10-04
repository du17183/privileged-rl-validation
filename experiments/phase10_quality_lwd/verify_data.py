"""Verify actual stored completed trajectories, terminal snapshots and counts."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from trajectory_quality.trajectory import read_completed
from trajectory_quality.quality_model import trajectory_score
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
path=ROOT/'datasets/phase10_quality_lwd'/f'{args.run}.h5'
marker=json.loads((ROOT/'checkpoints/phase10_quality_lwd'/args.run/'completed.json').read_text())
with h5py.File(path,'r') as h5:
    count=int(h5.attrs['complete_trajectories'])
    assert count==marker['online_episodes'] and len(h5['transitions/action'])==marker['steps']
    assert sum(h5['trajectories/success'][:])==marker['online_successes']
    total=int(np.sum(h5['trajectories/length'][:]))
    assert total+int(h5.attrs['partial_transitions'])==marker['steps']
for i in np.linspace(0,count-1,min(10,count)).astype(int) if count else []:
    trajectory=read_completed(path,int(i))
    assert trajectory['done'][-1,0]>.5 and not np.any(trajectory['done'][:-1]>.5)
    assert trajectory['state']['robot'].shape[1]==26 and trajectory['state']['privileged'].shape[1]==11
    assert trajectory['action'].shape[1]==7 and np.isfinite(trajectory['action']).all()
    recomputed=trajectory_score(trajectory['state']['privileged'],trajectory['next_state']['privileged'],trajectory['success'])
    assert abs(recomputed['quality_score']-trajectory['quality_score'])<1e-5
    assert abs(trajectory['reward'].sum()-trajectory['return_value'])<.001
result=dict(passed=True,run=args.run,completed=count,online_successes=marker['online_successes'],
            steps=marker['steps'],partial_transitions=marker['steps']-total,
            checks='index stride, unique terminal step, true terminal GT, quality reproducibility, return sums, field dimensions, count conservation')
(ROOT/'results/phase10_quality_lwd'/f'data_audit_{args.run}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
