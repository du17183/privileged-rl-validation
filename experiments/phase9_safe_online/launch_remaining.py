"""Adopt owned Phase 9 jobs and fill free slots; primary jobs are already started."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from experiments.phase9_safe_online.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'
TRAIN='experiments/phase9_safe_online/train.py'

def processes(script):
    records=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid() or Path(os.readlink(entry/'cwd')).resolve()!=ROOT:continue
            args=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if script in args:records.append((int(entry.name),args))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return records

def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def main():
    for pid,_ in processes('experiments/phase9_safe_online/launch.py'):
        os.kill(pid,signal.SIGTERM)
    time.sleep(1)
    lock=(OUT/'launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={}
    for pid,args in processes(TRAIN):
        if '--smoke' in args:continue
        arm,seed=args[args.index('--arm')+1],int(args[args.index('--seed')+1])
        entries=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
        gpu=int(next(v.split(b'=',1)[1] for v in entries if v.startswith(b'CUDA_VISIBLE_DEVICES=')))
        running[pid]=(None,arm,seed,gpu,'adopted',None)
    completed=set()
    for arm in ARMS:
        for seed in range(5):
            path=ROOT/'checkpoints/phase9_safe_online'/f'P9{arm}_seed{seed}'/'completed.json'
            if path.exists() and json.loads(path.read_text())['steps']==300000:completed.add((arm,seed))
    active={(r[1],r[2]) for r in running.values()}
    pending=[(arm,seed) for arm_group in [('A','B','C','D'),('CANN','D100','D30')] for seed in range(5) for arm in arm_group if (arm,seed) not in active|completed]
    (OUT/'queue_adoption.json').write_text(json.dumps(dict(adopted_pids=list(running),
        max_jobs_per_gpu=3,total_jobs=35,reason='All primary runs started; secondary jobs fill unused slots rather than waiting for a stage barrier',
        algorithm_and_budget_unchanged=True),indent=2))
    with (OUT/'launch_status_continued.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('arm','seed','gpu','pid','start_utc','end_utc','exit_code'));stream.flush()
        while pending or running:
            for pid,(proc,arm,seed,gpu,start,log) in list(running.items()):
                code=proc.poll() if proc else (None if alive(pid) else 0)
                if code is None:continue
                marker=ROOT/'checkpoints/phase9_safe_online'/f'P9{arm}_seed{seed}'/'completed.json'
                if code==0 and (not marker.exists() or json.loads(marker.read_text())['steps']!=300000):code=1
                writer.writerow((arm,seed,gpu,pid,start,datetime.now(timezone.utc).isoformat(),code));stream.flush()
                if log:log.close()
                del running[pid]
                print(f'finished {arm}/{seed} gpu{gpu} exit={code}',flush=True)
                if code:raise RuntimeError('Failed run; other launched jobs continue')
                completed.add((arm,seed))
            readings=[tuple(int(v.strip()) for v in line.split(',')) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
            counts={g:sum(r[3]==g for r in running.values()) for g,_,_ in readings}
            while pending:
                free=[r for r in readings if counts[r[0]]<3 and r[1]<80000]
                if not free:break
                gpu,_,_=min(free,key=lambda r:(counts[r[0]],r[2],r[1]))
                arm,seed=pending.pop(0);run=f'P9{arm}_seed{seed}'
                if (ROOT/'checkpoints/phase9_safe_online'/run).exists():raise RuntimeError('Unexpected existing run')
                log=(ROOT/'logs/phase9_safe_online'/f'{run}.log').open('x')
                proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',TRAIN,'--arm',arm,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                running[proc.pid]=(proc,arm,seed,gpu,datetime.now(timezone.utc).isoformat(),log)
                counts[gpu]+=1
                print(f'started {run} gpu{gpu} pid={proc.pid}',flush=True)
            if all((a,s) in completed for a in ('A','B','C','D') for s in range(5)) and not (OUT/'primary_completed.json').exists():
                (OUT/'primary_completed.json').write_text(json.dumps(dict(runs=20,steps=6000000)))
            time.sleep(5)
    (OUT/'training_completed.json').write_text(json.dumps(dict(runs=35,steps=10500000)))

if __name__=='__main__':main()
