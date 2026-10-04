"""Bounded own-job queue using available headroom; no external changes."""
import csv,fcntl,json,os,subprocess,time
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG,CKPT,VARIANTS,run_name
from experiments.phase12_environment_state_feedback.resources import idle_readings
def now():return datetime.now(timezone.utc).isoformat()
def main():
 lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not json.loads((OUT/'preflight.json').read_text())['passed']:raise RuntimeError('Preflight required')
 # All 20 core jobs are dispatched before the 30 strength jobs.
 pending=[(a,s,seed) for seed in range(5) for a,s in VARIANTS[:4]]+[(a,s,seed) for seed in range(5) for a,s in VARIANTS[4:]]
 running={};completed=[];cap=3;allowed=tuple(range(8))
 (OUT/'queue_config.json').write_text(json.dumps(dict(total_jobs=50,max_jobs_per_gpu=cap,gpus=allowed,
  training_envs=32,evaluator_envs=32,batch_size=256,updates_per_vector_step=4,total_training_steps=15000000,
  external_processes='GPU admission excludes every external compute PID',started_at=now()),indent=2))
 with (OUT/'launch_status.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=('arm','strength','seed','gpu','pid','start','end','exit_code'));writer.writeheader()
  while pending or running:
   for pid,(proc,record,log) in list(running.items()):
    code=proc.poll()
    if code is None:continue
    log.close();marker=CKPT/run_name(record['arm'],record['strength'],record['seed'])/'completed.json'
    if code==0 and (not marker.exists() or json.loads(marker.read_text())['steps']!=300000):code=1
    record.update(end=now(),exit_code=code);writer.writerow(record);f.flush();completed.append(record);del running[pid]
    print('finished',record,flush=True)
    if code:raise RuntimeError('Own training failed. Other processes untouched.')
   readings,blocked=idle_readings(running.keys())
   counts={g:sum(r[1]['gpu']==g for r in running.values()) for g,_,_ in readings}
   while pending:
    available=[r for r in readings if r[0] in allowed and counts[r[0]]<cap and r[1]<160000]
    if not available:break
    gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1],r[0]));arm,strength,seed=pending.pop(0)
    name=run_name(arm,strength,seed)
    if (CKPT/name).exists():raise RuntimeError('Refuse overwriting run')
    log=(LOG/f'{name}.log').open('x')
    proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',f'experiments/{OUT.name}/train.py','--arm',arm,'--strength',strength,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    record=dict(arm=arm,strength=strength,seed=seed,gpu=gpu,pid=proc.pid,start=now());running[proc.pid]=(proc,record,log);counts[gpu]+=1
    print('started',record,flush=True)
   temp=OUT/'queue_state.tmp';temp.write_text(json.dumps(dict(timestamp=now(),running=[v[1] for v in running.values()],pending=pending,completed=completed,blocked_by_external_compute_gpus=blocked),indent=2));os.replace(temp,OUT/'queue_state.json');time.sleep(5)
 (OUT/'training_completed.json').write_text(json.dumps(dict(runs=50,steps=15000000,completed_at=now()),indent=2))
if __name__=='__main__':main()
