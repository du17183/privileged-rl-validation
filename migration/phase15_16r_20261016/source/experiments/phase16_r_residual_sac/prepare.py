"""Read-only historical verification, followed by new protocol attestation."""
from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from experiments.phase16_r_residual_sac.common import ROOT,EXP,RES,sha,save,utc

def main():
    if (EXP/'protocol_lock.json').exists():raise RuntimeError('Preparation already frozen')
    issues=[];protected={};checks={}
    old=json.loads((ROOT/'phase15_3/results/phase15_2_baseline_audit.json').read_text())
    assert old['passed']
    for rel,expected in old['sha256'].items():
        if sha(ROOT/rel)!=expected:issues.append('phase15_2 preservation '+rel)
        protected[rel]=expected
    m=json.loads((ROOT/'phase15_3/results/delivery/manifest_20261009T030401Z.json').read_text())
    for rel,row in m['files'].items():
        if sha(ROOT/rel)!=row['sha256']:issues.append('phase15_3 preservation '+rel)
        protected[rel]=row['sha256']
    checks['historical_protected_files']=len(protected)
    selected=json.loads((ROOT/'results/phase14_3_pi05_bc/selection.json').read_text())
    declared=json.loads((ROOT/'phase15/results/normal_policy_manifest.json').read_text())
    normal={};rates=[]
    for seed in range(5):
        step=selected[f'C_seed{seed}']['step']
        rel=f'checkpoints/phase15/normal_policy/C_seed{seed}_step{step}.pt'
        path=ROOT/rel;expected=declared[rel]['sha256']
        assert sha(path)==sha(ROOT/declared[rel]['original'])==expected
        meta=torch.load(path,map_location='cpu',weights_only=False)
        assert meta['seed']==seed and meta['variant']=='C' and meta['update']==step and meta['mode']=='full'
        assert np.asarray(meta['mean']).shape==(39,) and np.asarray(meta['std']).shape==(39,)
        p=ROOT/f'results/phase14_3_pi05_bc/test/C_seed{seed}/random.json'
        result=json.loads(p.read_text());rows=list(csv.DictReader(p.with_suffix('.csv').open()))
        assert len(rows)==result['episodes']==128 and result['model_seed']==seed
        rate=np.mean([int(r['success']) for r in rows]);assert rate==result['success']
        rates.append(float(rate))
        normal[str(seed)]=dict(checkpoint=rel,sha256=expected,original=declared[rel]['original'],historical_random_success=float(rate),model_seed=seed,update=step,mean=np.asarray(meta['mean']).tolist(),std=np.asarray(meta['std']).tolist())
        for q in [path,ROOT/declared[rel]['original'],p,p.with_suffix('.csv')]:protected[str(q.relative_to(ROOT))]=sha(q)
    assert np.mean(rates)==.896875
    base=Path('/home/xiaolong/pi05_peg_lwd/checkpoints/pi05_base/model.safetensors')
    base_expected='0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'
    if sha(base)!=base_expected:issues.append('pretrained pi05 base changed')
    protected[str(base)]=base_expected
    data=ROOT/'datasets/phase14_2/prepared_v1/manifest.json'
    dm=json.loads(data.read_text());protected[str(data.relative_to(ROOT))]=sha(data)
    for rel,expected in dm['raw_sha256'].items():
        if sha(ROOT/rel)!=expected:issues.append('original expert data changed '+rel)
        protected[rel]=expected
    for p in data.parent.rglob('*'):
        if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
    for scope,split in dm['splits'].items():
        sets=[set(split[k]) for k in ['train','validation','test']]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    for folder in ['door_env','envs','randomized_env','environment_state','progress_rl','pi05','phase15_1/inference_opt']:
        for p in (ROOT/folder).rglob('*.py'):protected[str(p.relative_to(ROOT))]=sha(p)
    for rel in ['configs/runtime_env.sh','assets/panda_door_cabinet.usd','results/phase14_3_pi05_bc/selection.json','experiments/phase14_3_pi05_bc/protocol.json','phase15/results/normal_policy_manifest.json']:
        protected[rel]=sha(ROOT/rel)
    req=EXP/'user_request.txt';assert sha(req)=='e4a03804c88e350100c0ff5f0d4b99ce3e5ea2085861d03005aceabd741f262b'
    checks.update(five_seed_random=rates,historical_random_mean=float(np.mean(rates)),historical_data_splits_disjoint=True,normal_checkpoints_identical=True,base_sha256=base_expected,RL_expert_data=0)
    save(RES/'historical_inputs.json',dict(utc=utc(),passed=not issues,issues=issues,normal=normal,checks=checks,protected=protected,GT_fields=['robot26','door_angle','door_velocity','progress','remaining_angle','left_contact','right_contact','handle_pose7'],action_scale=[.05,.05,.05,.3,.3,.3],old_rates_not_new_baseline=True))
    if issues:raise RuntimeError(str(issues))
    save(EXP/'protocol_lock.json',dict(utc=utc(),sha256=sha(EXP/'protocol.json'),user_request_sha256=sha(req),frozen_before_new_physical_results=True))
    save(RES/'source_manifest.json',dict(utc=utc(),files={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/phase16_r_residual_sac','algorithms/residual_sac','evaluation/phase16_r'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt']}))
    print(json.dumps(dict(passed=True,checks=checks,protected_files=len(protected))),flush=True)

if __name__=='__main__':main()
