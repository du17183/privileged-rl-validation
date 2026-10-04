"""Resumable bounded BC-only pipeline, shared GPUs without preemption."""
import argparse,json,os,subprocess,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase14_2_recovery_bc';L=ROOT/'logs/phase14_2_recovery_bc';C=ROOT/'checkpoints/phase14_2_recovery_bc'
lock=threading.Lock();status={}

def mark(key,**values):
    with lock:
        status[key]=dict(utc=datetime.now(timezone.utc).isoformat(),**values)
        temp=R/'jobs.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(R/'jobs.json')
    print(json.dumps(dict(job=key,**values)),flush=True)

def run(key,module,args,gpu=None):
    e=os.environ.copy();e.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (L/(key+'.log')).open('a') as f:
        child=subprocess.Popen(['nice','-n','10',sys.executable,'-u','-m',module,*args],cwd=ROOT,env=e,
                               stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        mark(key,status='running',pid=child.pid,gpu=gpu)
        rc=child.wait()
    mark(key,status='complete' if rc==0 else 'failed',exit_code=rc)
    if rc:raise RuntimeError(key)

def wait_for(path,deadline=7200):
    begin=time.monotonic()
    while not path.exists():
        if time.monotonic()-begin>deadline:raise TimeoutError(path)
        time.sleep(5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-gpus',default='0,6');p.add_argument('--eval-gpus',default='0,2,3,6');a=p.parse_args()
    R.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
    wait_for(ROOT/'datasets/phase14_2/success_size_matched/collection_v1/summary.json')
    m=ROOT/'datasets/phase14_2/prepared_v1/manifest.json'
    if not m.exists():run('prepare','experiments.phase14_2_recovery_bc.prepare',[])
    slots=a.train_gpus.split(',')
    def fit(slot):
        for seed in range(slot,5,len(slots)):
            if (C/f'bc_seed{seed}_summary.json').exists():continue
            if (C/f'bc_seed{seed}.csv').exists():raise RuntimeError('Partial training retained, use a separate retry namespace')
            run(f'train_seed{seed}','experiments.phase14_2_recovery_bc.train',
                ['--data',str(m.parent),'--output',str(C),'--seed',str(seed),'--device','cuda:0'],slots[slot])
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for f in [pool.submit(fit,s) for s in range(len(slots))]:f.result()
    for case in ['ee_offset','contact_loss','door_regression','stagnation']:
        wait_for(R/'cohorts'/(case+'.json'))
    jobs=[]
    for seed in range(5):
        for arm in ['A','B','C','D','E','F','D_no_handle']:
            jobs.append(dict(checkpoint=f'checkpoints/phase14_2_recovery_bc/{arm}_seed{seed}_best.pt',
                             output=f'results/phase14_2_recovery_bc/evaluation/{arm}_seed{seed}'))
    # Verify that common validation selection does not hide a stronger S-only
    # baseline. All six arms also undergo matched random evaluation at20k.
    for seed in range(5):
        for arm in ['A','B','C','D','E','F']:
            jobs.append(dict(checkpoint=f'checkpoints/phase14_2_recovery_bc/{arm}_seed{seed}_final.pt',
                             output=f'results/phase14_2_recovery_bc/evaluation_final/{arm}_seed{seed}',tests=['random']))
    (R/'evaluation_jobs.json').write_text(json.dumps(jobs,indent=2));slots=a.eval_gpus.split(',')
    def evaluate(slot):
        run(f'eval_slot{slot}','experiments.phase14_2_recovery_bc.evaluate',
            ['--jobs',str(R/'evaluation_jobs.json'),'--slot',str(slot),'--slots',str(len(slots)),'--device','cuda:0'],slots[slot])
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for f in [pool.submit(evaluate,s) for s in range(len(slots))]:f.result()
    missing=[str(ROOT/j['output']/(t+'.json')) for j in jobs for t in j.get('tests',['fixed','random','ee_offset','contact_loss','door_regression','stagnation','natural_severe']) if not (ROOT/j['output']/(t+'.json')).exists()]
    if missing:raise RuntimeError('Simulator returned0 but missing expected results: '+str(missing[:5]))
    run('audit_after','experiments.phase14_2_recovery_bc.audit',['--check'])
    mark('pipeline',status='complete',rl_updates=0,trained_models=35)

if __name__=='__main__':main()
