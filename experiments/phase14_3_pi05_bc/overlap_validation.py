"""Use six freed GPUs for validation while the last two fits continue.

Formal evaluation locks prevent duplication with the later main pipeline.
No training/evaluation protocol changes; no other project's process touched.
"""
import json,os,subprocess,time
from .run import ROOT,R,L,ISAAC,PI,job,TESTS

def main():
    while True:
        if (R/'pipeline_failed.txt').exists():raise RuntimeError((R/'pipeline_failed.txt').read_text())
        tasks=json.loads((R/'lifecycle.json').read_text())
        occupied={v['gpu'] for k,v in tasks.items() if k.startswith('pi05_') and v['status']=='running'}
        free=[g for g in range(8) if g not in occupied]
        ready=all((ROOT/f'checkpoints/phase14_3_pi05_bc/{arm}_seed{seed}_complete.json').exists() for arm in ['C','D'] for seed in range(4))
        if ready and len(free)>=6:break
        time.sleep(15)
    free=free[:6];jobs=[job(a,s,k,'validation') for s in range(4) for a in ['C','D'] for k in [250,1250,2500,3750,5000]]
    destination=R/'overlap_validation_jobs.json';destination.write_text(json.dumps(jobs,indent=2));tasks=[]
    for slot,gpu in enumerate(free):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=str(gpu))
        sock=R/f'overlap_validation_rpc{slot}.sock'
        for kind,cmd in [('server',[PI,'-u','-m','pi05.inference.server','--socket',str(sock)]),
            ('eval',[ISAAC,'-u','-m','evaluation.pi05_closed_loop_eval','--jobs',str(destination),'--slot',str(slot),
              '--slots','6','--socket',str(sock),'--device','cuda:0'])]:
            with (L/f'overlap_validation_{kind}{slot}.log').open('a') as f:
                p=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            tasks.append((p,dict(kind=kind,slot=slot,gpu=gpu,pid=p.pid,status='running',started=time.time())))
    record=R/'overlap_validation_lifecycle.json'
    while any(p.poll() is None for p,d in tasks):
        for p,d in tasks:
            if p.poll() is not None:d.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode)
        record.write_text(json.dumps([d for p,d in tasks],indent=2))
        if any(p.poll() not in [None,0] for p,d in tasks):raise RuntimeError([d for p,d in tasks if p.poll() not in [None,0]])
        time.sleep(5)
    assert all((ROOT/j['output']/(t+'.json')).exists() for j in jobs for t in TESTS)
    (R/'overlap_validation_complete.json').write_text(json.dumps(dict(checkpoints=40,tests=280,gpus=free,completed=time.time())))
    print('OVERLAP VALIDATION COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (R/'overlap_validation_failed.txt').write_text(traceback.format_exc());raise
