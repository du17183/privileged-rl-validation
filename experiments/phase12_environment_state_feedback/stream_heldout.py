"""Run unchanged heldout tests of completed runs on GPUs with no other compute jobs."""
import csv,fcntl,json,os,subprocess,time,traceback
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG,CKPT,VARIANTS,run_name
from experiments.phase12_environment_state_feedback.resources import idle_readings

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)
def eligible(job):
 a,s,i=job;name=run_name(a,s,i);marker=CKPT/name/'completed.json'
 if not marker.exists():return False
 if json.loads(marker.read_text())['steps']!=300000:raise RuntimeError('Unexpected completed run budget')
 return all(p.exists() for p in (CKPT/name/'best.pt',CKPT/name/'step_300000.pt',ROOT/'datasets'/OUT.name/f'{name}.h5'))

def main():
 lock=(OUT/'heldout_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pending=[(a,s,i) for i in range(5) for a,s in VARIANTS];running={};completed=[]
 with (OUT/'heldout_status.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=('arm','strength','seed','gpu','pid','start','end','exit_code'));writer.writeheader()
  while pending or running:
   if list(OUT.glob('error_*')) or (OUT/'queue_continuation_error.json').exists():raise RuntimeError('Training failed')
   for pid,(proc,row,log) in list(running.items()):
    code=proc.poll()
    if code is None:continue
    log.close();marker=OUT/'heldout'/f"{run_name(row['arm'],row['strength'],row['seed'])}.json"
    if code==0 and not marker.exists():code=1
    row.update(end=now(),exit_code=code);writer.writerow(row);f.flush();completed.append(row);del running[pid]
    print('completed',row,flush=True)
    if code:raise RuntimeError('Owned heldout job failed')
   readings,blocked=idle_readings(running.keys())
   counts={g:sum(v[1]['gpu']==g for v in running.values()) for g,_,_ in readings}
   while pending:
    available=[r for r in readings if counts[r[0]]<6 and r[1]<160000]
    ready=next((job for job in pending if eligible(job)),None)
    if not available or ready is None:break
    gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1],r[0]));pending.remove(ready);a,s,i=ready;name=run_name(a,s,i)
    log=(LOG/f'heldout_{name}.log').open('x')
    proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',f'experiments/{OUT.name}/heldout.py',
     '--arm',a,'--strength',s,'--seed',str(i),'--device','cuda:0'],cwd=ROOT,
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    row=dict(arm=a,strength=s,seed=i,gpu=gpu,pid=proc.pid,start=now());running[proc.pid]=(proc,row,log);counts[gpu]+=1
    print('started',row,flush=True)
   atomic(OUT/'heldout_queue_state.json',dict(timestamp=now(),running=[v[1] for v in running.values()],pending=pending,
    eligible_pending=[job for job in pending if eligible(job)],completed=completed,blocked_by_external_compute_gpus=blocked,
    supervisor='stream_heldout',max_jobs_per_gpu=6,original_evaluator=True))
   time.sleep(5)
 results=[json.loads((OUT/'heldout'/f"{run_name(r['arm'],r['strength'],r['seed'])}.json").read_text()) for r in completed]
 atomic(OUT/'heldout_completed.json',dict(jobs=len(results),episodes=sum(r['total_episodes'] for r in results),
  interactions=sum(r['total_interactions'] for r in results),completed_at=now(),scheduler='stream_heldout'))
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(OUT/'heldout_scheduler_error.json',dict(traceback=traceback.format_exc()));raise
