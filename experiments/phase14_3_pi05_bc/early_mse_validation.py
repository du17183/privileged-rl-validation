"""Evaluate already-finished MSE models while pretrained models are fitting.

Uses exactly the formal jobs/outputs. Own lifecycle file avoids overwriting
the training supervisor. This process must finish before main validation.
"""
import json,os,subprocess,time
from pathlib import Path
from .run import ROOT,R,L,ISAAC,job,TESTS

def main():
    jobs=[job(a,s,k,'validation') for s in range(5) for a in ['A','B'] for k in [1000,5000,10000,15000,20000]]
    destination=R/'early_mse_validation_jobs.json'
    destination.write_text(json.dumps(jobs,indent=2))
    record=R/'early_mse_validation_lifecycle.json';procs=[];info=[]
    for slot in range(8):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=str(slot))
        with (L/f'early_mse_validation_{slot}.log').open('a') as f:
            p=subprocess.Popen(['nice','-n','10',ISAAC,'-u','-m','evaluation.pi05_closed_loop_eval','--jobs',str(destination),
                '--slot',str(slot),'--slots','8','--device','cuda:0'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        procs.append(p);info.append(dict(slot=slot,pid=p.pid,status='running',started=time.time()))
    record.write_text(json.dumps(info,indent=2))
    while any(p.poll() is None for p in procs):
        for p,d in zip(procs,info):
            if p.poll() is not None:d.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode)
        record.write_text(json.dumps(info,indent=2));time.sleep(5)
    assert all(p.returncode==0 for p in procs),info
    assert all((ROOT/j['output']/(t+'.json')).exists() for j in jobs for t in TESTS)
    (R/'early_mse_validation_complete.json').write_text(json.dumps(dict(checkpoints=50,tests=350,completed=time.time())))
    print('EARLY MSE VALIDATION COMPLETE',flush=True)

if __name__=='__main__':main()
