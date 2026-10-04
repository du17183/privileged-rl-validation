"""Independent evaluation queue; starts only when each checkpoint is finalized."""
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from experiments.phase9_safe_online.protocol import ARMS
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/phase9_safe_online'

def main():
    # Wait until all training jobs have been dispatched: their remaining
    # concurrency only decreases, so idle slots can safely be used for tests.
    while not all((ROOT/'checkpoints/phase9_safe_online'/f'P9{a}_seed{s}').exists() for a in ARMS for s in range(5)):
        if list(OUT.glob('error_*.txt')):raise RuntimeError('Training failed')
        time.sleep(10)
    if not (OUT/'heldout_pilot/C_seed0.json').exists():
        raise RuntimeError('Physical perturbation pilot gate missing')
    jobs = [(arm,seed) for seed in range(5) for arm in ARMS]
    running = {}
    with (OUT/'heldout_status.csv').open('x',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('arm','seed','gpu','pid','exit_code'))
        while jobs or running:
            for pid,(proc,arm,seed,gpu,log) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                path = OUT/'heldout'/f'{arm}_seed{seed}.json'
                expected = (25 if arm in ('A','C') else 22) if arm in ('A','B','C','D') else (9 if arm=='CANN' else 6)
                if code==0 and (not path.exists() or len(json.loads(path.read_text()))!=expected):
                    code=1
                writer.writerow((arm,seed,gpu,pid,code))
                stream.flush()
                log.close()
                del running[pid]
                print(f'Heldout {arm}/{seed} exit={code}',flush=True)
                if code:
                    raise RuntimeError('Independent test failed; preserve outputs')
            readings = [tuple(int(v.strip()) for v in s.split(',')) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
            counts = {g:sum(r[3]==g for r in running.values()) for g,_ in readings}
            while jobs:
                training={g:0 for g,_ in readings}
                for entry in Path('/proc').iterdir():
                    if not entry.name.isdigit():continue
                    try:
                        if entry.stat().st_uid!=os.getuid() or Path(os.readlink(entry/'cwd')).resolve()!=ROOT:continue
                        arguments=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
                        if 'experiments/phase9_safe_online/train.py' not in arguments or '--smoke' in arguments:continue
                        environment=(entry/'environ').read_bytes().split(b'\0')
                        gpu=int(next(v.split(b'=',1)[1] for v in environment if v.startswith(b'CUDA_VISIBLE_DEVICES=')))
                        training[gpu]+=1
                    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
                free = [(g,m) for g,m in readings if counts[g]+training[g]<3 and m<80000]
                if not free:
                    break
                eligible=[(i,a,s) for i,(a,s) in enumerate(jobs) if (ROOT/'checkpoints/phase9_safe_online'/f'P9{a}_seed{s}'/'completed.json').exists()]
                if not eligible:break
                gpu,_ = min(free,key=lambda r:(counts[r[0]]+training[r[0]],r[1]))
                index,arm,seed=eligible[0];jobs.pop(index)
                log = (ROOT/'logs/phase9_safe_online'/f'heldout_{arm}_seed{seed}.log').open('x')
                proc = subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','experiments/phase9_safe_online/evaluate_frozen.py',
                    '--arm',arm,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                running[proc.pid]=(proc,arm,seed,gpu,log)
                counts[gpu]+=1
            if list(OUT.glob('error_*.txt')):raise RuntimeError('Training failed')
            time.sleep(5)
    (OUT/'heldout_completed.json').write_text(json.dumps(dict(runs=35)))

if __name__=='__main__':
    main()
