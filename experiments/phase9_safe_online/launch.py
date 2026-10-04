"""Bounded Phase 9 queue with validated completion markers and immutable sources."""
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from experiments.phase9_safe_online.protocol import PROTOCOL
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/phase9_safe_online'
LOG = ROOT/'logs/phase9_safe_online'

def main():
    lock = (OUT/'launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (OUT/'pilot_completed.json').exists():
        raise RuntimeError('Pilot completion gate missing')
    status = OUT/'launch_status.csv'
    if status.exists():
        raise RuntimeError('Existing queue; inspect rather than duplicate launch')
    snapshot = OUT/'training_source'
    snapshot.mkdir(exist_ok=False)
    manifest = {}
    files = list((ROOT/'safe_online').glob('*.py'))+list((ROOT/'experiments/phase9_safe_online').glob('*.py'))+[ROOT/'replay/success_buffer.py']
    for path in files:
        relative = path.relative_to(ROOT)
        destination = snapshot/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'training_source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    stages = [('primary',('A','B','C','D')),('secondary',('CANN','D100','D30'))]
    with status.open('x',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('stage','arm','seed','gpu','pid','start_utc','end_utc','exit_code'))
        stream.flush()
        for stage, arms in stages:
            pending = [(a,s) for s in range(5) for a in arms]
            running = {}
            while pending or running:
                for pid, record in list(running.items()):
                    proc, arm, seed, gpu, start, log = record
                    code = proc.poll()
                    if code is None:
                        continue
                    marker = ROOT/'checkpoints/phase9_safe_online'/f'P9{arm}_seed{seed}'/'completed.json'
                    if code==0:
                        if not marker.exists():
                            code = 1
                        else:
                            data = json.loads(marker.read_text())
                            if data['steps']!=300000 or data['optimizer_steps']!=37500:
                                code = 1
                    writer.writerow((stage,arm,seed,gpu,pid,start,datetime.now(timezone.utc).isoformat(),code))
                    stream.flush()
                    log.close()
                    del running[pid]
                    print(f'finished {arm} seed{seed} gpu{gpu} exit={code}',flush=True)
                    if code:
                        raise RuntimeError(f'Run failed: {arm}/{seed}; other launched runs continue; no silent overwrite')
                query = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                readings = [tuple(int(v.strip()) for v in line.split(',')) for line in query.splitlines()]
                counts = {g:sum(r[3]==g for r in running.values()) for g,_,_ in readings}
                while pending:
                    free = [(g,m,u) for g,m,u in readings if counts[g]<3 and m<80000]
                    if not free:
                        break
                    gpu,_,_ = min(free,key=lambda r:(counts[r[0]],r[2],r[1]))
                    arm,seed = pending.pop(0)
                    run = f'P9{arm}_seed{seed}'
                    if (ROOT/'checkpoints/phase9_safe_online'/run).exists():
                        raise RuntimeError(f'Unexpected existing run {run}')
                    log = (LOG/f'{run}.log').open('x')
                    proc = subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','experiments/phase9_safe_online/train.py',
                        '--arm',arm,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                    start = datetime.now(timezone.utc).isoformat()
                    running[proc.pid] = (proc,arm,seed,gpu,start,log)
                    counts[gpu] += 1
                    print(f'started {run} gpu{gpu} pid={proc.pid}',flush=True)
                time.sleep(5)
            (OUT/f'{stage}_completed.json').write_text(json.dumps(dict(arms=arms,seeds=5,steps=300000)))
    (OUT/'training_completed.json').write_text(json.dumps(dict(runs=35,steps=10500000)))

if __name__=='__main__':
    main()
