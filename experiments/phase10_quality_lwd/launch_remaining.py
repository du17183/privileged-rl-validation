"""Adopt unchanged owned learners; increase concurrency without more samples."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
TRAIN='experiments/phase10_quality_lwd/train.py'


def now():return datetime.now(timezone.utc).isoformat()


def processes(token):
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid() or Path(os.readlink(entry/'cwd')).resolve()!=ROOT:continue
            command=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if token in command:found.append((int(entry.name),command))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    for pid,_ in processes('experiments.phase10_quality_lwd.launch'):
        os.kill(pid,signal.SIGTERM)
    time.sleep(1)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=json.loads((OUT/'queue_state.json').read_text())
    start_lookup={(r['arm'],r['seed']):r['start'] for r in original['running']}
    running={};completed=[]
    for pid,command in processes(TRAIN):
        if '--smoke' in command:continue
        arm=command[command.index('--arm')+1];seed=int(command[command.index('--seed')+1])
        entries=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
        gpu=int(next(e.split(b'=',1)[1] for e in entries if e.startswith(b'CUDA_VISIBLE_DEVICES=')))
        record=dict(arm=arm,seed=seed,gpu=gpu,pid=pid,start=start_lookup[(arm,seed)],adopted=True)
        running[pid]=(None,record,None)
    for arm in ARMS:
        for seed in range(5):
            marker=ROOT/'checkpoints/phase10_quality_lwd'/f'P10{arm}_seed{seed}'/'completed.json'
            if marker.exists():
                if json.loads(marker.read_text())['steps']!=300000:raise RuntimeError('Bad formal budget')
                completed.append(dict(arm=arm,seed=seed))
    active={(r[1]['arm'],r[1]['seed']) for r in running.values()}
    done={(r['arm'],r['seed']) for r in completed}
    pending=[(arm,seed) for seed in range(5) for arm in ARMS if (arm,seed) not in active|done]
    control=OUT/'concurrency_control.json'
    with control.open('x') as f:json.dump(dict(max_jobs_per_gpu=3),f)
    (OUT/'queue_adoption.json').write_text(json.dumps(dict(adopted_pids=list(running),total_jobs=35,
        old_max_jobs_per_gpu=2,new_max_jobs_per_gpu=3,training_envs_per_job=96,
        all_learning_hyperparameters_unchanged=True,reason='User throughput preference; successful first cohort and 33GB of275GB GPU memory',
        external_jobs_untouched=True,adopted_at=now()),indent=2))
    with (OUT/'launch_status_continued.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','seed','gpu','pid','start','end','exit_code','adopted']);writer.writeheader()
        while pending or running:
            for pid,(proc,record,log) in list(running.items()):
                code=proc.poll() if proc else (None if alive(pid) else 0)
                if code is None:continue
                marker=ROOT/'checkpoints/phase10_quality_lwd'/f"P10{record['arm']}_seed{record['seed']}"/'completed.json'
                if code==0 and (not marker.exists() or json.loads(marker.read_text())['steps']!=300000):code=1
                if log:log.close()
                record.update(end=now(),exit_code=code);writer.writerow(record);stream.flush()
                del running[pid]
                print('finished',record,flush=True)
                if code:raise RuntimeError('Owned run failed; no external processes modified')
                completed.append(dict(arm=record['arm'],seed=record['seed']))
            cap=int(json.loads(control.read_text())['max_jobs_per_gpu'])
            if cap not in (1,2,3):raise RuntimeError('Invalid bounded concurrency')
            readings=[tuple(int(v.strip()) for v in row.split(',')) for row in subprocess.check_output([
                'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
            counts={g:sum(r[1]['gpu']==g for r in running.values()) for g,_,_ in readings}
            while pending:
                available=[r for r in readings if counts[r[0]]<cap and r[1]<80000]
                if not available:break
                gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1]));arm,seed=pending.pop(0)
                name=f'P10{arm}_seed{seed}'
                if (ROOT/'checkpoints/phase10_quality_lwd'/name).exists():raise RuntimeError('Unowned/exited partial run; preserve for diagnosis')
                log=(ROOT/'logs/phase10_quality_lwd'/f'{name}.log').open('x')
                proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',TRAIN,'--arm',arm,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                record=dict(arm=arm,seed=seed,gpu=gpu,pid=proc.pid,start=now(),adopted=False)
                running[proc.pid]=(proc,record,log);counts[gpu]+=1;print('started',record,flush=True)
            tmp=OUT/'queue_state.tmp';tmp.write_text(json.dumps(dict(timestamp=now(),max_jobs_per_gpu=cap,
                running=[r[1] for r in running.values()],pending=pending,completed=completed),indent=2));os.replace(tmp,OUT/'queue_state.json')
            time.sleep(5)
    (OUT/'training_completed.json').write_text(json.dumps(dict(runs=35,steps=10500000,completed_at=now()),indent=2))

if __name__=='__main__':main()
