"""Audit complete Phase 3 budgets, heldout pairing, and runtime provenance."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'/'door_privileged_ablation'
CHECKPOINTS=ROOT/'checkpoints'/'door_privileged_ablation'
VARIANTS=('diag_A','diag_B','B1','B2','B5','B6','E0','E1','E2','E3')
SOURCES=('assets/panda_door_cabinet.usd','door_env/door.py','door_env/isaac_env.py',
         'door_dataset/door_expert_1000.h5','algorithms/asymmetric_sac.py',
         'algorithms/door_phase3_sac.py','auxiliary_learning/gt_prediction.py',
         'experiments/door_privileged_ablation/train.py',
         'configs/door_experiment.json','configs/door_phase3.json')


def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block=stream.read(8*1024*1024)
            if not block: break
            sha.update(block)
    return sha.hexdigest()


def version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def main():
    count=0
    for variant in VARIANTS:
        for seed in range(5):
            run=f'{variant}_seed{seed}'
            curve=rows(RESULTS/f'eval_{run}.csv')
            if len(curve)!=21 or int(curve[0]['env_steps'])!=0 or int(curve[-1]['env_steps'])!=500000:
                raise AssertionError(f'Incomplete evaluation curve: {run}')
            if any(int(a['env_steps'])>=int(b['env_steps']) for a,b in zip(curve,curve[1:])):
                raise AssertionError(f'Non-monotonic steps: {run}')
            diag=RESULTS/f'diagnostics_{run}.csv'
            with diag.open(newline='',encoding='utf-8') as stream:
                telemetry=list(csv.DictReader(stream))
            if len(telemetry)!=15625 or int(telemetry[-1]['optimizer_steps'])!=62500:
                raise AssertionError(f'Incomplete optimizer telemetry: {run}')
            checkpoint=CHECKPOINTS/run/'step_500000.pt'
            if not checkpoint.is_file(): raise AssertionError(f'Missing final checkpoint: {run}')
            best=max(curve,key=lambda r:float(r['success_rate']))
            for mode in ('best','final'):
                evaluation=rows(RESULTS/f'heldout_{run}_{mode}.csv')
                expected=int(best['env_steps']) if mode=='best' else 500000
                if len(evaluation)!=1 or int(evaluation[0]['episodes'])!=64 or \
                   int(evaluation[0]['selected_steps'])!=expected:
                    raise AssertionError(f'Invalid heldout pairing: {run} {mode}')
            count+=1
    selected=json.loads((CHECKPOINTS/'selected'/'E2'/'selection_manifest.json').read_text(encoding='utf-8'))
    if sum(item['accepted'] and Path(item['actor_checkpoint']).is_file()
           for item in selected['selected_actors'])!=5:
        raise AssertionError('Expected five validated E2 actor exports')
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                                   text=True).splitlines()[0].strip()
    report={'protocol':'door_phase3_privileged_ablation_v1',
            'training_variants':list(VARIANTS),'seeds':[0,1,2,3,4],
            'complete_training_runs':count,'online_steps_per_run':500000,
            'new_online_training_interactions':count*500000,
            'gradient_updates_per_run':62500,
            'training_eval_points_per_run':21,'training_eval_episodes_per_point':64,
            'independent_heldout_checkpoint_replays':count*2,
            'independent_heldout_episodes_per_replay':64,
            'validated_robot_only_E2_actors':5,
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                       'numpy':np.__version__,'h5py':h5py.__version__,
                       'isaaclab':version('isaaclab'),'isaaclab_tasks':version('isaaclab_tasks'),
                       'gpu_name':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),
                       'nvidia_driver':driver},
            'sha256':{name:digest(ROOT/name) for name in SOURCES}}
    path=RESULTS/'verification.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('complete_training_runs',
                     'new_online_training_interactions','independent_heldout_checkpoint_replays',
                     'validated_robot_only_E2_actors','runtime')},indent=2))


if __name__=='__main__':
    main()
