"""Same physical evaluation as14.2, new independent validation/test cohorts."""
import argparse,os,sys,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--slots',type=int,default=8)
p.add_argument('--socket');AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app
import fcntl,json,time
from pathlib import Path
import numpy as np,torch,isaaclab_tasks
from randomized_env.door_randomization import create
from experiments.phase14_2_recovery_bc.model import load
from experiments.phase14_3_pi05_bc.ordinary_eval import ordinary
from evaluation.recovery_policy_eval import pressure
from pi05.inference import PolicyClient
ROOT=Path(__file__).resolve().parents[1]

def evaluate_job(env,job,client):
    checkpoint=ROOT/job['checkpoint'];out=ROOT/job['output'];out.mkdir(parents=True,exist_ok=True)
    # Pipeline overlap changes scheduling only. File lock prevents a main
    # worker and an overlap worker from rerunning/writing the same evaluation.
    with (out/'.evaluation.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        meta=torch.load(checkpoint,map_location='cpu',weights_only=False)
        is_pi=meta['variant'] in ['C','D'];mean=torch.tensor(meta['mean'],device=env.device);std=torch.tensor(meta['std'],device=env.device)
        if is_pi:
            if client is None:client=PolicyClient(a.socket,meta['mean'],meta['std'])
            meta=client.load(checkpoint);client.ablation=job.get('ablation');net=client
        else:net,mean,std,meta=load(checkpoint,env.device)
        for test in job['tests']:
            path=out/(test+'.json')
            if path.exists():continue
            start=time.monotonic()
            if test in ['fixed','random']:
                if is_pi:client.configure();client.seed=job['seed']+300
                result=ordinary(env,net,mean,std,meta,0 if test=='fixed' else 2,job['fixed_n'] if test=='fixed' else job['random_n'],path,job['seed'])
            else:
                cohort=ROOT/job['cohorts']/(test+'.h5');env.set_randomization(2)
                if is_pi:client.configure(cohort);client.seed=job['seed']+300
                result=pressure(env,net,mean,std,meta,cohort,path)
            result.update(elapsed_s=time.monotonic()-start,checkpoint=str(checkpoint),protocol='phase14_3',selection_split=job['split'],
                          action_horizon=10 if is_pi else 1,execute_steps=5 if is_pi else 1,controller_unchanged=True)
            if is_pi:result['inference']=client.stats()
            path.write_text(json.dumps(result,indent=2));print('COMPLETE '+json.dumps(dict(arm=meta['variant'],seed=meta['seed'],update=meta['update'],test=test,success=result['success'],elapsed_s=result['elapsed_s'])),flush=True)
    return client

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if Path(a.jobs).name=='validation_jobs.json':
        early=ROOT/'results/phase14_3_pi05_bc'
        while (early/'early_mse_validation_lifecycle.json').exists() and not (early/'early_mse_validation_complete.json').exists():
            errors=list(early.glob('early_mse_validation_jobs.json.slot*.failed'))
            if errors:raise RuntimeError(errors[0].read_text())
            time.sleep(5)
    env=create(32,a.device or 'cuda:0',143501,level=2);client=None
    jobs=json.loads(Path(a.jobs).read_text())
    try:
        for job in jobs[a.slot::a.slots]:client=evaluate_job(env,job,client)
    finally:
        if client is not None:client.stop()
        env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc();print(error,flush=True);Path(a.jobs+f'.slot{a.slot}.failed').write_text(error)
        os._exit(1)
    # Isaac5.1 app.close previously left completed workers spinning. Files are
    # flushed/closed by main; normal process exit reclaims this own GPU context.
    print('EVALUATOR COMPLETE',flush=True);os._exit(0)
