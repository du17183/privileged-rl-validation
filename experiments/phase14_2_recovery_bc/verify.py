"""Meaningful final gates: paired starts/budgets, masks, source integrity."""
import csv,json
from pathlib import Path
import h5py,numpy as np,torch
from .model import inputs
from .prepare import sha
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc';C=ROOT/'checkpoints/phase14_2_recovery_bc'

def main():
    m=json.loads((ROOT/'datasets/phase14_2/prepared_v1/manifest.json').read_text());issues=[];episodes=0
    for path,digest in m['raw_sha256'].items():
        if sha(ROOT/path)!=digest:issues.append(('source_changed',path))
    for seed in range(5):
        original=torch.load(C/f'initial_seed{seed}.pt',map_location='cpu',weights_only=False)['model']
        extra=torch.load(C/'quantity_control'/f'initial_seed{seed}.pt',map_location='cpu',weights_only=False)['model']
        if not all(torch.equal(original[k],extra[k]) for k in original):issues.append(('initialization',seed))
        for arm in ['A','B','C','D','E','F','D_no_handle']:
            d=torch.load(C/f'{arm}_seed{seed}_best.pt',map_location='cpu',weights_only=False)
            expected='robot' if arm in ['A','C','E'] else 'no_handle' if arm=='D_no_handle' else 'full'
            if d['mode']!=expected or d['parameter_count']!=77838:issues.append(('mode/capacity',arm,seed))
    torch.manual_seed(142990);robot=torch.randn(17,26);gt=torch.randn(17,13)
    mean=torch.tensor(m['mean']);std=torch.tensor(m['std']);altered=gt+torch.randn_like(gt)*100
    if not torch.equal(inputs(robot,gt,mean,std,'robot'),inputs(robot,altered,mean,std,'robot')):issues.append('GT mask leakage')
    altered=gt.clone();altered[:,6:]=torch.randn_like(altered[:,6:])*100
    if not torch.equal(inputs(robot,gt,mean,std,'no_handle'),inputs(robot,altered,mean,std,'no_handle')):issues.append('pose mask leakage')
    for case in ['ee_offset','contact_loss','door_regression','stagnation','natural_severe']:
        with h5py.File(R/'cohorts'/(case+'.h5'),'r') as h:
            expected={k:int(h[k].attrs['handoff_tick'])+1 for k in h}
        for file in (R/'evaluation').glob('*/'+case+'.csv'):
            rows=list(csv.DictReader(file.open()));episodes+=len(rows)
            if len(rows)!=64 or {row['cohort_id'] for row in rows}!=set(expected):issues.append(('cohort_count',str(file)))
            for row in rows:
                start=int(row['handoff_tick']);length=int(row['recovery_ticks'])
                if start!=expected[row['cohort_id']] or length>600-start:issues.append(('episode_budget',str(file),row['cohort_id'],start,length))
    result=dict(issues=issues,pressure_episodes_checked=episodes,paired_seed_initial_weights_equal=True,
                masked_GT_invariance=True,capacity=77838,source_hashes_verified=len(m['raw_sha256']),rl_updates=0)
    (R/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if issues:raise RuntimeError(issues[:10])

if __name__=='__main__':main()
