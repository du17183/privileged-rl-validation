import csv,json
import numpy as np,torch
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase14_2_recovery_bc.model import inputs,measured

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

