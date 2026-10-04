"""Audit Phase 4 completeness, preserved inputs, and trajectory schema."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stable_privileged_rl'
VARIANTS=('E0','E25','E50','E100','E200','E100RB','E100R1','E100M','Q0','QGT','QV')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def verify_preserved(manifest):
    count=0
    for line in (ROOT/manifest).read_text().splitlines():
        if not line.strip(): continue
        expected,name=line.split(maxsplit=1)
        name=name.lstrip('*')
        if digest(ROOT/name)!=expected:
            raise RuntimeError(f'Preserved input changed: {name}')
        count+=1
    return count


def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    preserved=verify_preserved('results/door/manifest_sha256.txt')+verify_preserved('results/primary_manifest_sha256.txt')
    expected_fields=('state','action','reward','next_state','return','value','success','quality_score')
    initial_pairs=[]
    for seed in range(5):
        ref=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'E0_seed{seed}'/'step_0.pt',
                       map_location='cpu',weights_only=False)
        for variant in ('E25','E50','E100','E200','E100RB','E100R1'):
            alt=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'{variant}_seed{seed}'/'step_0.pt',
                           map_location='cpu',weights_only=False)
            if not all(torch.equal(v,alt['actor'][k]) for k,v in ref['actor'].items()):
                raise RuntimeError(f'Unpaired actor initialization: {variant} seed {seed}')
            initial_pairs.append(f'{variant}_seed{seed}')
        multi=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'E100M_seed{seed}'/'step_0.pt',
                         map_location='cpu',weights_only=False)
        if not all(torch.equal(v,multi['actor'][k]) for k,v in ref['actor'].items()
                   if k.startswith('encoder.') or k.startswith('policy_head.')):
            raise RuntimeError(f'Unpaired multi-task policy initialization: seed {seed}')
        initial_pairs.append(f'E100M_seed{seed}')
        qref=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'Q0_seed{seed}'/'step_0.pt',
                        map_location='cpu',weights_only=False)
        for variant in ('QGT','QV'):
            alt=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'{variant}_seed{seed}'/'step_0.pt',
                           map_location='cpu',weights_only=False)
            if not all(torch.equal(v,alt['actor'][k]) for k,v in qref['actor'].items()):
                raise RuntimeError(f'Unpaired shared-value initialization: {variant} seed {seed}')
            initial_pairs.append(f'{variant}_seed{seed}')
        for left,right,step in (('E50','E100',50016),('E100','E200',100000)):
            first=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'{left}_seed{seed}'/f'step_{step}.pt',
                             map_location='cpu',weights_only=False)
            second=torch.load(ROOT/'checkpoints'/'stable_privileged_rl'/f'{right}_seed{seed}'/f'step_{step}.pt',
                              map_location='cpu',weights_only=False)
            if not all(torch.equal(v,second['actor'][k]) for k,v in first['actor'].items()):
                raise RuntimeError(f'Unpaired pre-cutoff actor: {left}/{right} seed {seed} step {step}')
            initial_pairs.append(f'{left}_{right}_seed{seed}_step{step}')
    for variant in VARIANTS:
        for seed in range(5):
            run=f'{variant}_seed{seed}'
            evaluation=rows(OUT/f'eval_{run}.csv')
            if len(evaluation)!=51 or int(evaluation[-1]['env_steps'])!=500000:
                raise RuntimeError(f'Incomplete evaluation: {run}')
            diagnostics=rows(OUT/f'diagnostics_{run}.csv')
            if len(diagnostics)!=15625 or not any(
                    int(r['episodes_in_window'])>0 for r in diagnostics):
                raise RuntimeError(f'Incomplete diagnostics: {run}')
            for mode in ('best','final'):
                heldout=rows(OUT/f'heldout_{run}_{mode}.csv')
                if len(heldout)!=1 or int(heldout[0]['episodes'])!=64:
                    raise RuntimeError(f'Incomplete heldout: {run}/{mode}')
            if variant in ('Q0','QGT','QV'):
                with h5py.File(OUT/f'trajectories_{run}.h5') as h5:
                    if (len(h5['action'])!=500000 or not all(k in h5 for k in expected_fields)
                            or int(h5.attrs['complete_episodes'])<100):
                        raise RuntimeError(f'Incomplete trajectory schema: {run}')
    sources=('auxiliary_learning/multitask_encoder.py','checkpoint_manager/rollback.py',
             'replay/value_weighted_replay_v2.py','experiments/stable_privileged_rl/train.py',
             'experiments/stable_privileged_rl/eval_client.py',
             'experiments/stable_privileged_rl/eval_worker.py',
             'experiments/stable_privileged_rl/analyze.py',
             'diagnostics/q_value_phase4.py',
             'diagnostics/trajectory_value_phase4.py',
             'diagnostics/representation_probe_phase4.py',
             'configs/door_phase4.json')
    result={'protocol':'door_phase4_stable_privileged_v2_independent_evaluation',
            'preserved_manifest_entries':preserved,
            'complete_training_runs':55,'online_steps_per_run':500000,
            'new_online_training_interactions':27500000,
            'training_eval_points_per_run':51,'training_eval_episodes_per_point':32,
            'independent_heldout_checkpoint_replays':110,
            'independent_heldout_episodes_per_replay':64,
            'trajectory_hdf5_files':15,
            'paired_initial_actor_checks':len(initial_pairs),
            'sha256':{name:digest(ROOT/name) for name in sources}}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
