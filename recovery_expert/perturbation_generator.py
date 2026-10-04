"""Measured action perturbations; only door regression explicitly writes the door."""
import math
import numpy as np
import torch
from door_env.door import door_index


class PerturbationGenerator:
    POLICY,INJECT,RECOVERY=range(3)
    def __init__(self,n,device,kind,seed,amplitudes=(.005,.01,.02)):
        self.n=n;self.device=device;self.kind=kind;self.seed=seed
        self.amplitudes=amplitudes
        self.stage=torch.zeros(n,device=device,dtype=torch.long)
        self.ticks=self.stage.clone();self.no_progress=self.stage.clone()
        self.reference_ee=torch.zeros((n,3),device=device)
        self.reference_angle=torch.zeros(n,device=device)
        self.actual_offset=self.reference_ee.clone();self.target_offset=self.reference_ee.clone()
        self.axis=torch.zeros(n,device=device,dtype=torch.long)
        self.amplitude=torch.zeros(n,device=device)
        self.had_contact=torch.zeros(n,device=device,dtype=torch.bool)
        self.valid=self.had_contact.clone();self.explicit_door_write=self.had_contact.clone()
        self.enabled=torch.ones(n,device=device,dtype=torch.bool)
        self.last_angle=torch.zeros(n,device=device)
        self.assignment=np.zeros(n,dtype=np.int64)

    def reset(self,ids,assignments):
        for tensor in [self.stage,self.ticks,self.no_progress,self.reference_angle,self.actual_offset,
                       self.target_offset,self.had_contact,self.valid,self.explicit_door_write,self.last_angle]:tensor[ids]=0
        for i,assignment in zip(ids,assignments):
            self.assignment[i]=assignment
            # Balanced Cartesian axes, signs and three amplitudes.
            cell=int(assignment)%18;axis=cell//6;amp=(cell%6)//2;sign=1 if cell%2 else -1
            self.axis[i]=axis;self.amplitude[i]=self.amplitudes[amp]
            self.target_offset[i,axis]=sign*self.amplitudes[amp]

    @torch.no_grad()
    def action(self,env,robot,gt,policy_action):
        action=policy_action.clone();age=env.episode_length_buf
        distance=(robot[:,18:21]-gt[:,2:5]).norm(dim=-1);contact=(gt[:,9:11]>.5).all(-1)
        idle=(self.stage==self.POLICY)&self.enabled
        if self.kind=='ee_offset':ready=idle&(age>30)&(distance<.115)&(distance>.045)&~contact
        elif self.kind=='contact_loss':ready=idle&contact&(gt[:,0]>.12)
        elif self.kind=='door_regression':ready=idle&contact&(gt[:,0]>math.radians(30))
        else:ready=idle&contact&(gt[:,0]>.15)&(gt[:,0]<.9)
        ids=ready.nonzero().flatten()
        self.stage[ids]=self.INJECT;self.ticks[ids]=0
        self.reference_ee[ids]=robot[ids,18:21];self.reference_angle[ids]=gt[ids,0]
        self.had_contact[ids]=contact[ids];self.last_angle[ids]=gt[ids,0]
        if self.kind=='door_regression' and len(ids):
            cabinet=env.scene['cabinet'];j=door_index(env)
            q=cabinet.data.joint_pos[ids].clone();v=cabinet.data.joint_vel[ids].clone()
            q[:,j]=math.radians(20);v[:,j]=0
            cabinet.write_joint_state_to_sim(q,v,env_ids=ids)
            self.explicit_door_write[ids]=True
        injecting=self.stage==self.INJECT
        if self.kind=='ee_offset':
            target=self.reference_ee+self.target_offset
            action[injecting,:3]=((target-robot[:,18:21])/.05).clamp(-.3,.3)[injecting]
            action[injecting,3:6]=0;action[injecting,6]=1
        elif self.kind=='contact_loss':
            action[injecting,:6]=0;action[injecting,6]=1
        elif self.kind=='door_regression':
            action[injecting,:6]=0;action[injecting,6]=1
        else:
            # Original controller receives nonzero rotation commands while the
            # tool remains at its current handle position. No joint lock/reward change.
            action[injecting,:6]=0;action[injecting,3]=.008;action[injecting,6]=-1
            flat=(gt[:,0]-self.last_angle).abs()<.002
            self.no_progress=torch.where(injecting&flat,self.no_progress+1,torch.zeros_like(self.no_progress))
            self.last_angle[injecting]=gt[injecting,0]
        self.ticks[injecting]+=1
        self.actual_offset=robot[:,18:21]-self.reference_ee
        minimum={'ee_offset':8,'contact_loss':16,'door_regression':3,'stagnation':30}[self.kind]
        if self.kind=='ee_offset':
            error=(self.actual_offset-self.target_offset).norm(dim=-1)
            criterion=error<torch.maximum(self.amplitude*.25,torch.full_like(self.amplitude,.0015))
            finished=injecting&(self.ticks>=minimum)&(criterion|(self.ticks>=48))
        elif self.kind=='contact_loss':
            criterion=self.had_contact&~contact;finished=injecting&(self.ticks>=minimum)
        elif self.kind=='door_regression':
            criterion=(self.reference_angle-gt[:,0])>math.radians(8)
            finished=injecting&(self.ticks>=minimum)
        else:
            criterion=(self.no_progress>=24)&((gt[:,0]-self.reference_angle).abs()<.025)
            finished=injecting&((criterion&(self.ticks>=minimum))|(self.ticks>=60))
        self.valid[finished]=criterion[finished];self.stage[finished]=self.RECOVERY
        return action,finished,self.valid
