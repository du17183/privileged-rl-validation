"""Bounded queue; only own Phase 10 processes and new output paths."""
import csv
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase10_quality_lwd'


def now():return datetime.now(timezone.utc).isoformat()


def main():
    lock=(OUT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pilot=ROOT/'checkpoints/phase10_quality_lwd/P10D10_seed0_smoke/completed.json'
    if not pilot.exists():raise RuntimeError('Pilot must finish before formal release')
    if not json.loads((OUT/'replay_audit.json').read_text())['passed']:raise RuntimeError('Replay audit failed')
    pending=[(arm,seed) for seed in range(5) for arm in ARMS]
    running={}
    records=[]
    cap=2
    (OUT/'queue_config.json').write_text(json.dumps(dict(total_jobs=35,max_jobs_per_gpu=cap,
        training_envs_per_job=96,paired_evaluator_envs=32,omp_threads=4,
        external_jobs='Existing RLinf jobs left running; no external processes controlled',
        pilot_steps=10080,formal_training_steps=10500000),indent=2))
    with (OUT/'launch_status.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','seed','gpu','pid','start','end','exit_code'])
        writer.writeheader()
        while pending or running:
            for pid,(proc,record,log) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                log.close()
                marker=ROOT/'checkpoints/phase10_quality_lwd'/f"P10{record['arm']}_seed{record['seed']}"/'completed.json'
                if code==0 and (not marker.exists() or json.loads(marker.read_text())['steps']!=300000):code=1
                record.update(end=now(),exit_code=code)
                writer.writerow(record);stream.flush()
                records.append(record)
                del running[pid]
                print('finished',record,flush=True)
                if code:raise RuntimeError('Owned run failed; other owned/external jobs untouched')
            readings=[tuple(int(v.strip()) for v in line.split(',')) for line in subprocess.check_output([
                'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
            counts={g:sum(r[1]['gpu']==g for r in running.values()) for g,_,_ in readings}
            while pending:
                available=[r for r in readings if counts[r[0]]<cap and r[1]<80000]
                if not available:break
                gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1]))
                arm,seed=pending.pop(0)
                name=f'P10{arm}_seed{seed}'
                if (ROOT/'checkpoints/phase10_quality_lwd'/name).exists():raise RuntimeError('Refuse overwriting run')
                log=(ROOT/'logs/phase10_quality_lwd'/f'{name}.log').open('x')
                proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','experiments/phase10_quality_lwd/train.py',
                    '--arm',arm,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                    stdout=log,stderr=subprocess.STDOUT)
                record=dict(arm=arm,seed=seed,gpu=gpu,pid=proc.pid,start=now())
                running[proc.pid]=(proc,record,log)
                counts[gpu]+=1
                print('started',record,flush=True)
            temp=OUT/'queue_state.tmp'
            temp.write_text(json.dumps(dict(timestamp=now(),running=[r[1] for r in running.values()],pending=pending,completed=records),indent=2))
            os.replace(temp,OUT/'queue_state.json')
            time.sleep(5)
    (OUT/'training_completed.json').write_text(json.dumps(dict(runs=35,steps=10500000,completed_at=now()),indent=2))

if __name__=='__main__':main()
