"""One simulator process evaluates several checkpoints, amortizing startup."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--slot',type=int,required=True)
p.add_argument('--slots',type=int,default=4);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app
import json,csv,time
from pathlib import Path
import numpy as np,torch,isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from .model import load,inputs,measured
from evaluation.recovery_policy_eval import pressure
ROOT=Path(__file__).resolve().parents[2]

def ordinary(env,net,mean,std,meta,level,n,path,seed=142501):
    env.set_randomization(level);torch.manual_seed(seed);env.reset(seed=seed)
    count=np.zeros(32,int);quota=np.full(32,n//32);quota[:n%32]+=1
    peak=torch.zeros(32,device=env.device);contacts=torch.zeros(32,device=env.device,dtype=torch.bool)
    rows=[];ticks=0
    while np.any(count<quota):
        with torch.no_grad():action=net(inputs(robot_observation(env),measured(env),mean,std,meta['mode']))
        _,_,term,trunc,_=env.step(action);_,gt,done=transition_after_step(env,term,trunc)
        peak=torch.maximum(peak,gt[:,0]);contacts|=(gt[:,9:11]>.5).all(-1)
        for i in done.nonzero().flatten().tolist():
            if count[i]<quota[i]:
                rows.append(dict(clone=i,episode=int(count[i]),success=int(gt[i,0]>1),max_angle_rad=float(peak[i]),
                                 final_angle_rad=float(gt[i,0]),contact_success=int(contacts[i]),
                                 angle0_rad=float(env.final_parameters[i,0]),offset_x=float(env.final_parameters[i,1]),
                                 offset_y=float(env.final_parameters[i,2]),offset_z=float(env.final_parameters[i,3])))
                count[i]+=1
            peak[i]=0;contacts[i]=False
        ticks+=1
    with path.with_suffix('.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(variant=meta['variant'],model_seed=meta['seed'],episodes=n,level=level,evaluation_seed=seed,interactions=ticks*32,
                checkpoint_update=meta['update'],mode=meta['mode'],**{k:float(np.mean([r[k] for r in rows])) for k in ['success','max_angle_rad','final_angle_rad','contact_success']})
    path.write_text(json.dumps(result,indent=2));return result

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    jobs=json.loads(Path(a.jobs).read_text());env=create(32,a.device or 'cuda:0',142501,level=2)
    try:
        for job in jobs[a.slot::a.slots]:
            net,mean,std,meta=load(ROOT/job['checkpoint'],env.device);out=ROOT/job['output'];out.mkdir(parents=True,exist_ok=True)
            tests=job.get('tests',['fixed','random','ee_offset','contact_loss','door_regression','stagnation','natural_severe'])
            for test in tests:
                path=out/(test+'.json')
                if path.exists():continue
                begin=time.monotonic()
                if test in ['fixed','random']:result=ordinary(env,net,mean,std,meta,0 if test=='fixed' else 2,64 if test=='fixed' else 128,path,job.get('seed',142501))
                else:
                    marker=ROOT/'results/phase14_2_recovery_bc/cohorts'/(test+'.json')
                    deadline=time.monotonic()+7200
                    while not marker.exists():
                        if time.monotonic()>deadline:raise TimeoutError(marker)
                        time.sleep(5)
                    env.set_randomization(2)
                    result=pressure(env,net,mean,std,meta,ROOT/'results/phase14_2_recovery_bc/cohorts'/(test+'.h5'),path)
                result['elapsed_s']=time.monotonic()-begin;path.write_text(json.dumps(result,indent=2))
                print('COMPLETE '+json.dumps(dict(checkpoint=job['checkpoint'],test=test,**result)),flush=True)
    finally:env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback,sys
        traceback.print_exc();sys.stderr.flush()
        Path(a.jobs+'.slot'+str(a.slot)+'.failed').write_text(traceback.format_exc())
        raise
    finally:app.close()
