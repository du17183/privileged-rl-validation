"""Queue independent tests after training; bounded two tests per GPU."""
import json
import os
import subprocess
import time
from pathlib import Path
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase10_quality_lwd'


def main():
    if not (OUT/'training_completed.json').exists():raise RuntimeError('Training incomplete')
    pending=[];finished=[]
    for seed in range(5):
        for arm in ARMS:
            output=OUT/'heldout'/f'{arm}_seed{seed}.json'
            if output.exists():
                rows=json.loads(output.read_text())
                q=json.loads((OUT/'heldout'/f'q_diagnosis_{arm}_seed{seed}.json').read_text())
                if not rows or len(q)!=64:raise RuntimeError('Incomplete published test; preserve and diagnose')
                finished.append(dict(arm=arm,seed=seed))
            else:pending.append((arm,seed))
    running={}
    while pending or running:
        for pid,(proc,arm,seed,gpu,log) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            log.close()
            destination=OUT/'heldout'/f'{arm}_seed{seed}.json'
            if code or not destination.exists():raise RuntimeError(f'Heldout {arm}/{seed} failed; other jobs untouched')
            finished.append(dict(arm=arm,seed=seed));del running[pid]
            print('heldout finished',arm,seed,flush=True)
        counts={gpu:sum(v[3]==gpu for v in running.values()) for gpu in range(8)}
        while pending:
            available=[g for g in range(8) if counts[g]<2]
            if not available:break
            gpu=min(available,key=lambda g:counts[g]);arm,seed=pending.pop(0)
            log=(ROOT/'logs/phase10_quality_lwd'/f'heldout_{arm}_seed{seed}.log').open('a')
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','experiments/phase10_quality_lwd/evaluate_frozen.py',
                '--arm',arm,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                stdout=log,stderr=subprocess.STDOUT)
            running[proc.pid]=(proc,arm,seed,gpu,log);counts[gpu]+=1
        (OUT/'heldout_queue_state.json').write_text(json.dumps(dict(finished=finished,pending=pending,
            running=[dict(pid=pid,arm=r[1],seed=r[2],gpu=r[3]) for pid,r in running.items()]),indent=2))
        time.sleep(5)
    rows=[]
    for record in finished:rows+=json.loads((OUT/'heldout'/f"{record['arm']}_seed{record['seed']}.json").read_text())
    (OUT/'heldout_completed.json').write_text(json.dumps(dict(jobs=35,tests=len(rows),episodes=sum(r['episodes'] for r in rows),
        evaluation_steps=sum(r['eval_env_steps'] for r in rows)),indent=2))

if __name__=='__main__':main()
