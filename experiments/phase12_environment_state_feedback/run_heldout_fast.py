import csv,json,os,subprocess,time
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG,VARIANTS,run_name
from experiments.phase12_environment_state_feedback.resources import idle_readings
def now():return datetime.now(timezone.utc).isoformat()
def main():
 if not json.loads((OUT/'throughput/concurrency_preflight.json').read_text())['passed']:raise RuntimeError('Concurrency throughput preflight failed')
 if not (OUT/'training_completed.json').exists():raise RuntimeError('Training incomplete')
 pending=[(a,s,i) for i in range(5) for a,s in VARIANTS];running={};completed=[]
 with (OUT/'heldout_status.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=('arm','strength','seed','gpu','pid','start','end','exit_code'));w.writeheader()
  while pending or running:
   for pid,(proc,r,log) in list(running.items()):
    code=proc.poll()
    if code is None:continue
    log.close();marker=OUT/'heldout'/f"{run_name(r['arm'],r['strength'],r['seed'])}.json"
    if code==0 and not marker.exists():code=1
    r.update(end=now(),exit_code=code);w.writerow(r);f.flush();completed.append(r);del running[pid]
    if code:raise RuntimeError('Owned heldout job failed')
   readings,blocked=idle_readings(running.keys())
   counts={g:sum(v[1]['gpu']==g for v in running.values()) for g,_,_ in readings}
   while pending:
    available=[r for r in readings if counts[r[0]]<6 and r[1]<160000]
    if not available:break
    gpu,_,_=min(available,key=lambda r:(counts[r[0]],r[2],r[1],r[0]));a,s,i=pending.pop(0);name=run_name(a,s,i)
    log=(LOG/f'heldout_{name}.log').open('x');proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',f'experiments/{OUT.name}/heldout.py','--arm',a,'--strength',s,'--seed',str(i),'--device','cuda:0'],cwd=ROOT,
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    r=dict(arm=a,strength=s,seed=i,gpu=gpu,pid=proc.pid,start=now());running[proc.pid]=(proc,r,log);counts[gpu]+=1
   temp=OUT/'heldout_queue_state.tmp';temp.write_text(json.dumps(dict(timestamp=now(),running=[v[1] for v in running.values()],pending=pending,completed=completed,blocked_by_external_compute_gpus=blocked),indent=2));os.replace(temp,OUT/'heldout_queue_state.json');time.sleep(5)
 results=[json.loads((OUT/'heldout'/f"{run_name(r['arm'],r['strength'],r['seed'])}.json").read_text()) for r in completed]
 (OUT/'heldout_completed.json').write_text(json.dumps(dict(jobs=len(results),episodes=sum(r['total_episodes'] for r in results),interactions=sum(r['total_interactions'] for r in results),completed_at=now()),indent=2))
if __name__=='__main__':main()
