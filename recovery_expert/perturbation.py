"""Physical recovery tests: expert preparation then bounded action disturbance."""
import torch
from experiments.phase13_random_expert_bc.planner import RandomDoorPlanner


class Perturbation:
    def __init__(self,n,device,dt,kind):
        self.kind=kind;self.planner=RandomDoorPlanner(n,device,dt,release_orientation=True)
        self.stage=torch.zeros(n,device=device,dtype=torch.long)
        self.ticks=self.stage.clone();self.start_angle=torch.zeros(n,device=device)
        self.start_distance=self.start_angle.clone();self.valid=torch.zeros(n,device=device,dtype=torch.bool)
        self.had_contact=self.valid.clone()
        self.start_ee=torch.zeros((n,3),device=device)

    def reset(self,ids):
        self.planner.reset(ids)
        for t in [self.stage,self.ticks,self.start_angle,self.start_distance,self.valid,self.had_contact]:t[ids]=0

    @torch.no_grad()
    def action(self,env,robot,gt):
        distance=torch.linalg.vector_norm(robot[:,18:21]-gt[:,2:5],dim=-1)
        grasp=(gt[:,9:11]>.5).all(-1)
        ready=(self.stage==0)&grasp&(gt[:,0]>(.55 if self.kind=='door_regression' else .25))
        self.stage[ready]=1;self.ticks[ready]=0;self.start_angle[ready]=gt[ready,0]
        self.start_distance[ready]=distance[ready];self.had_contact[ready]=grasp[ready]
        self.start_ee[ready]=robot[ready,18:21]
        action=self.planner.action(env)
        disturb=self.stage==1;action[disturb,:6]=0;action[disturb,6]=1
        normal=torch.stack((gt[:,0].cos(),gt[:,0].sin(),torch.zeros_like(gt[:,0])),-1)
        if self.kind=='ee_offset':
            moving=disturb&(self.ticks>=16)
            target=self.start_ee[moving].clone();target[:,2]+=.05
            action[moving,:3]=((target-robot[moving,18:21])/.05).clamp(-.5,.5)
        elif self.kind=='door_regression':action[disturb,:3]=-.06*normal[disturb]
        self.ticks[disturb]+=1
        target={'contact_loss':16,'ee_offset':40,'door_regression':36}[self.kind]
        finished=disturb&(self.ticks>=target)
        if self.kind=='contact_loss':criteria=self.had_contact&~grasp
        elif self.kind=='ee_offset':criteria=(distance-self.start_distance)>.02
        else:criteria=(self.start_angle-gt[:,0])>.08
        self.valid[finished]=criteria[finished];self.stage[finished]=2
        return action,finished,self.valid
