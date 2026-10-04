import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, LOG, ARMS
p=argparse.ArgumentParser();p.add_argument('--conditional', action='store_true');args=p.parse_args()
def now(): return datetime.now(timezone.utc).isoformat()


def main():
    if args.conditional:
        if not json.loads((OUT/'conditional_gate.json').read_text())['passed']: raise RuntimeError('Conditional gate not met')
        pending = [('D', s, False) for s in range(5)]
    else:
        if not (OUT/'training_completed.json').exists(): raise RuntimeError('Training incomplete')
        pending = [(a, s, False) for s in range(5) for a in ARMS]+[('A', s, True) for s in range(5)]
    tag = 'conditional' if args.conditional else 'heldout';completed = [];running = {}
    with (OUT/f'{tag}_status.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('arm', 'seed', 'frozen', 'gpu', 'pid', 'start', 'end', 'exit_code'));writer.writeheader()
        while pending or running:
            for pid, (proc, record, log) in list(running.items()):
                code=proc.poll()
                if code is None: continue
                log.close();name=f"{record['arm']}_seed{record['seed']}"+('_frozen' if record['frozen'] else '')
                marker=OUT/tag/f'{name}.json'
                if code==0 and not marker.exists(): code=1
                record.update(end=now(), exit_code=code);writer.writerow(record);stream.flush();completed.append(record);del running[pid]
                if code: raise RuntimeError('Owned heldout job failed; no other jobs changed')
            readings=[tuple(int(v.strip()) for v in line.split(',')) for line in subprocess.check_output([
                'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
            counts={g:sum(r[1]['gpu']==g for r in running.values()) for g,_,_ in readings}
            while pending:
                available=[r for r in readings if counts[r[0]]<3 and r[1]<80000]
                if not available: break
                gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1],r[0]))
                arm,seed,frozen=pending.pop(0);name=f'{arm}_seed{seed}'+('_frozen' if frozen else '')
                log=(LOG/f'{tag}_{name}.log').open('x')
                command=[str(ROOT/'.venv/bin/python'),'-u','experiments/phase11_parameter_generalization/heldout.py','--arm',arm,'--seed',str(seed),'--device','cuda:0']
                if frozen: command.append('--frozen')
                if args.conditional: command.append('--conditional')
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                record=dict(arm=arm,seed=seed,frozen=frozen,gpu=gpu,pid=proc.pid,start=now());running[proc.pid]=(proc,record,log);counts[gpu]+=1
            temp=OUT/f'{tag}_queue_state.tmp';temp.write_text(json.dumps(dict(timestamp=now(),running=[v[1] for v in running.values()],pending=pending,completed=completed),indent=2));os.replace(temp,OUT/f'{tag}_queue_state.json')
            time.sleep(5)
    results=[json.loads((OUT/tag/(f"{r['arm']}_seed{r['seed']}"+('_frozen' if r['frozen'] else '')+'.json')).read_text()) for r in completed]
    (OUT/f'{tag}_completed.json').write_text(json.dumps(dict(jobs=len(completed),completed_at=now(),episodes=sum(r['total_episodes'] for r in results),
        interactions=sum(r['total_interactions'] for r in results)),indent=2))
if __name__=='__main__': main()
