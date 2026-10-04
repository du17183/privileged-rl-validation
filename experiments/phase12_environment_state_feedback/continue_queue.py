"""Adopt live owned jobs, start the last two, preserve all learning state.

Only the obsolete queue supervisor is terminated; trainers/evaluators continue.
The original frozen training/evaluation code is never edited.
"""
import csv,fcntl,json,os,signal,subprocess,time
from datetime import datetime,timezone
import psutil
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG,CKPT,VARIANTS,run_name
from experiments.phase12_environment_state_feedback.resources import idle_readings
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,path)
def main():
 old_pid=1038627
 old=psutil.Process(old_pid)
 if old.cmdline()[-2:]!=['-m','experiments.phase12_environment_state_feedback.launch']:
  raise RuntimeError('Unexpected supervisor identity')
 if (OUT/'training_completed.json').exists():raise RuntimeError('Already complete')
 snapshot=json.loads((OUT/'queue_state.json').read_text())
 atomic(OUT/'queue_before_throughput_amendment.json',snapshot)
 old.terminate();old.wait(timeout=15)
 lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 records={run_name(r['arm'],r['strength'],r['seed']):r for r in snapshot['running']+snapshot['completed']}
 running={};completed=[];started=set()
 for proc in psutil.process_iter(['pid','cmdline','create_time']):
  cmd=proc.info['cmdline'] or []
  if 'experiments/phase12_environment_state_feedback/train.py' not in cmd:continue
  if proc.cwd()!=str(ROOT):raise RuntimeError('Unexpected owned trainer path')
  a=cmd[cmd.index('--arm')+1];s=cmd[cmd.index('--strength')+1];seed=int(cmd[cmd.index('--seed')+1]);name=run_name(a,s,seed)
  if (a,s) not in VARIANTS or seed not in range(5):raise RuntimeError('Unexpected run')
  gpu=int(proc.environ()['CUDA_VISIBLE_DEVICES'])
  record=records.get(name,dict(arm=a,strength=s,seed=seed,gpu=gpu,pid=proc.pid,start=datetime.fromtimestamp(proc.create_time(),timezone.utc).isoformat()))
  running[proc.pid]=(proc,record,None,proc.create_time());started.add(name)
 for a,s in VARIANTS:
  for seed in range(5):
   name=run_name(a,s,seed);marker=CKPT/name/'completed.json'
   if name in started:continue
   if marker.exists():
    r=records.get(name,dict(arm=a,strength=s,seed=seed,start='recovered'))
    completed.append(dict(**r));started.add(name)
 pending=[(a,s,seed) for seed in range(5) for a,s in VARIANTS if run_name(a,s,seed) not in started]
 if len(pending)!=2:raise RuntimeError(f'Expected only two unstarted jobs: {pending}')
 atomic(OUT/'queue_config_continued.json',dict(max_jobs_per_gpu=4,total_jobs=50,initial_running=len(running),initial_completed=len(completed),
    initially_pending=pending,amendment='Only concurrency; same 32env,256batch,4updates,300k,seed,algorithm,episode budgets',started_at=now()))
 with (OUT/'launch_status_continued.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=('arm','strength','seed','gpu','pid','start','end','exit_code'));w.writeheader()
  while pending or running:
   for pid,(proc,r,log,created) in list(running.items()):
    if isinstance(proc,subprocess.Popen):alive=proc.poll() is None
    else:
     try:alive=proc.is_running() and proc.create_time()==created and proc.status()!=psutil.STATUS_ZOMBIE
     except psutil.NoSuchProcess:alive=False
    if alive:continue
    if log is not None:log.close()
    marker=CKPT/run_name(r['arm'],r['strength'],r['seed'])/'completed.json'
    code=0 if marker.exists() and json.loads(marker.read_text())['steps']==300000 else 1
    r.update(end=now(),exit_code=code);w.writerow(r);f.flush();completed.append(r);del running[pid];print('finished',r,flush=True)
    if code:raise RuntimeError('Owned job failed; no trainer restarted')
   readings,blocked=idle_readings(running.keys())
   counts={g:sum(v[1]['gpu']==g for v in running.values()) for g,_,_ in readings}
   while pending:
    available=[r for r in readings if counts[r[0]]<4 and r[1]<160000]
    if not available:break
    gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1],r[0]));a,s,seed=pending.pop(0);name=run_name(a,s,seed)
    if (CKPT/name).exists():raise RuntimeError('Refuse overwriting any run')
    log=(LOG/f'{name}.log').open('x')
    proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',f'experiments/{OUT.name}/train.py','--arm',a,'--strength',s,'--seed',str(seed),'--device','cuda:0'],cwd=ROOT,
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    r=dict(arm=a,strength=s,seed=seed,gpu=gpu,pid=proc.pid,start=now());running[proc.pid]=(proc,r,log,None);counts[gpu]+=1;print('started',r,flush=True)
   atomic(OUT/'queue_state.json',dict(timestamp=now(),running=[v[1] for v in running.values()],pending=pending,completed=completed,
       blocked_by_external_compute_gpus=blocked,supervisor='continue_queue',max_jobs_per_gpu=4));time.sleep(5)
 for a,s in VARIANTS:
  for seed in range(5):assert json.loads((CKPT/run_name(a,s,seed)/'completed.json').read_text())['steps']==300000
 atomic(OUT/'training_completed.json',dict(runs=50,steps=15000000,completed_at=now(),scheduler_amendment='queue_config_continued.json'))
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  atomic(OUT/'queue_continuation_error.json',dict(traceback=traceback.format_exc()));raise
