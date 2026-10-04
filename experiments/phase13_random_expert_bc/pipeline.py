"""Sequential, one-GPU Phase 13 stages with explicit evidence gates.

No outside process is stopped. Only one child uses the GPU at a time.
The RL stage is intentionally absent until the BC anchor gate is reviewed.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/phase13_random_expert_bc'
LOG=ROOT/'logs/phase13_random_expert_bc'
DATA=ROOT/'datasets/random_door_expert'
CHECK=ROOT/'checkpoints/phase13_random_expert_bc'


def state(stage,**kwargs):
    value=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kwargs)
    target=RESULT/'pipeline_state.json';temp=target.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(target)
    print(json.dumps(value),flush=True)


def run(name,module,arguments,timeout=3600):
    state(name,status='running')
    path=LOG/(name+'.log')
    if path.exists():raise FileExistsError(path)
    with path.open('x') as log:
        result=subprocess.run([sys.executable,'-u','-m',module,*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; {path}')
    state(name,status='complete')


def main():
    for path in [RESULT,LOG,DATA,CHECK]:path.mkdir(parents=True,exist_ok=True)
    if (RESULT/'pipeline_state.json').exists():raise FileExistsError('Pipeline already started')
    protocol=json.loads((ROOT/'experiments/phase13_random_expert_bc/protocol.json').read_text())
    (RESULT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (RESULT/'planner_amendment.json').write_text(json.dumps(dict(
        rationale='Legacy pilot 57/64; fully fixed-orientation staged pilot 0/64; released-opening-orientation pilot 62/64.',
        chosen='staged_free',pilot_seed=13001,independent_validation_seed=13101,
        unchanged=['task','reward','physics','reset','robot','existing differential IK'],
        gate_unchanged='unselected independent expert success >90%'),indent=2))
    run('expert_validation','experiments.phase13_random_expert_bc.expert',
        ['--mode','validate','--episodes',256,'--num-envs',32,'--seed',13101,'--planner','staged_free',
         '--output',RESULT/'expert_validation','--device','cuda:0'],1800)
    expert=json.loads((RESULT/'expert_validation/summary.json').read_text())
    if expert['success_rate']<=.9:
        state('expert_gate',status='failed',success=expert['success_rate']);return
    state('expert_gate',status='passed',success=expert['success_rate'])
    run('expert_collection','experiments.phase13_random_expert_bc.expert',
        ['--mode','collect','--episodes',300,'--num-envs',32,'--max-attempts',600,'--seed',13201,
         '--planner','staged_free','--output',DATA/'collection_v1','--device','cuda:0'],1800)
    run('data_prepare','experiments.phase13_random_expert_bc.prepare',
        ['--dataset',DATA/'collection_v1/trajectories.h5','--validation',RESULT/'expert_validation/summary.json',
         '--output',DATA/'split_v1'],600)
    for seed in range(5):
        run(f'bc_seed{seed}','experiments.phase13_random_expert_bc.bc',
            ['--data',DATA/'split_v1','--output',CHECK,'--seed',seed,'--device','cuda:0'],3600)
    # Fresh Isaac process for each arm avoids persistent simulator history.
    for seed in range(5):
        for arm in ['A','B']:
            run(f'heldout_{arm}_seed{seed}','experiments.phase13_random_expert_bc.evaluate',
                ['--checkpoint',CHECK/f'{arm}_seed{seed}_best.pt','--output',RESULT/f'heldout/{arm}_seed{seed}_best.json',
                 '--episodes',128,'--num-envs',32,'--seed',13501+seed,'--device','cuda:0'],1800)
    state('bc_primary_complete',status='complete',next='paired analysis, diagnostics, and conditional anchor gate; no RL started')


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('failed',status='failed',error=repr(error));raise
