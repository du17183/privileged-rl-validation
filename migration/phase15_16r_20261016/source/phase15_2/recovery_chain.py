"""No physical state writes; all stages execute original seven-dimensional actions."""
import time
import numpy as np
import torch
import pinocchio as pin
from isaaclab.utils.math import compute_pose_error
from recovery_expert.ik_feasibility import project
from phase15.reacquire.gt_reacquire import GTReacquire as Original
from phase15_2.pregrasp.pregrasp_geometry import PregraspPlanner,rotation
from phase15_2.safe_retreat.retreat_v2 import retreat_command
from phase15_1.severe_reacquire.regrasp import valid_grasp


class Reacquire(Original):
    def __init__(self,env,variant='B',retreat='R2',geometry=False,strict=False):
        # Reuse only original storage/action schema, with new isolated planner.
        self.env=env;self.variant=variant;self.strict=strict;self.planner=PregraspPlanner(env,retreat=retreat,geometry=geometry)
        self.ready=np.zeros(32,bool);self.phase=np.full(32,'idle',object)
        self.path=[None]*32;self.cursor=np.zeros(32,int);self.elapsed=np.zeros(32,int)
        self.grasp_run=np.zeros(32,int);self.history=[[] for _ in range(32)];self.current=[None]*32
        self.grasp_target=[None]*32;self.retreat_run=np.zeros(32,int);self.stage_ticks=np.zeros(32,int)

    def reset(self,ids):
        super().reset(ids);self.retreat_run[ids]=0;self.stage_ticks[ids]=0

    def start(self,ids):
        for i in ids:
            self.planner.retreat_origins.pop(int(i),None);self.ready[i]=False;self.elapsed[i]=0;self.stage_ticks[i]=0;self.retreat_run[i]=0;self.grasp_run[i]=0
            meta=dict(trigger_tick=int(self.env.episode_length_buf[i]),variant=self.variant,
                ik_success=False,planning_success=False,reached_pregrasp=False,regrasp=False,handback=False,
                plan_s=0.,planning_not_attempted=True,
                safe_retreat=False,failure_stage=None,retreat_checks=[],geometry='unchanged frozen URDF/cabinet OBB collision model',
                original_episode_budget_ticks=600,no_state_writes=True,responsible_for_opening=False)
            self.history[i].append(meta);self.current[i]=meta;self.phase[i]='release';self.path[i]=None

    @torch.no_grad()
    def action(self,active):
        wall=time.perf_counter();env=self.env;term=env.action_manager.get_term('arm_action')
        p,q=term._compute_frame_pose();jac=term._compute_frame_jacobian().clone()
        damping=term._ik_controller.cfg.ik_params['lambda_val']
        inverse=torch.linalg.solve(jac@jac.transpose(1,2)+damping**2*torch.eye(6,device=env.device),jac).transpose(1,2)
        scale=term._scale
        mapping=(inverse*scale[:,None,:] if scale.ndim==2 else inverse*scale).cpu().numpy()
        joint=env.scene['robot'].data.joint_pos.cpu().numpy();gt=env.get_tool_state().cpu().numpy()
        action=torch.zeros((32,7),device=env.device);action[:,6]=1
        for i in np.flatnonzero(active):
            self.elapsed[i]+=1;self.stage_ticks[i]+=1;meta=self.current[i];phase=self.phase[i]
            if phase in ['failed','ready']:
                action[i,6]=-1 if phase=='ready' else 1;continue
            if self.elapsed[i]>min(330,600-meta['trigger_tick']-1):
                self.phase[i]='failed';meta['failure_stage']='time_budget';continue
            if phase=='release':
                if joint[i,7:9].sum()>.068 and not np.all(gt[i,9:11]>.5):
                    self.phase[i]='retreat';self.stage_ticks[i]=0
                elif self.stage_ticks[i]>25:
                    self.phase[i]='failed';meta['failure_stage']='release_contact_not_cleared'
                continue
            if phase=='retreat':
                sc=scale[i].cpu().numpy() if scale.ndim==2 else scale.cpu().numpy()
                u,safe,check=retreat_command(self.planner,int(i),joint[i],mapping[i],sc)
                if self.stage_ticks[i]%10==1:meta['retreat_checks'].append(check)
                self.retreat_run[i]=self.retreat_run[i]+1 if safe else 0
                if self.retreat_run[i]>=3:
                    meta.update(safe_retreat=True,retreat_tick=int(env.episode_length_buf[i]))
                    path,plan=self.planner.plan(int(i),self.planner.goal(int(i),True),151501+int(i)*1009+meta['trigger_tick'])
                    meta.update(plan);meta['planning_not_attempted']=False;meta['failure_stage']=plan.get('failure')
                    self.path[i]=path;self.cursor[i]=1;self.phase[i]='pregrasp' if path is not None else 'failed';self.stage_ticks[i]=0
                elif u is not None:action[i,:6]=torch.tensor(u,device=env.device)
                else:self.phase[i]='failed';meta['failure_stage']='safe_retreat_no_monotone_escape'
                continue
            if phase=='pregrasp':
                path=self.path[i];k=self.cursor[i];desired=path[k]
                goal=self.planner.fk(desired);actual=self.planner.fk(joint[i])
                reached=np.linalg.norm(goal.translation-actual.translation)<.007 and np.linalg.norm(pin.log3(goal.rotation@actual.rotation.T))<.07
                if reached:
                    k+=1;self.cursor[i]=k
                    if k==len(path):
                        self.phase[i]='align';meta.update(reached_pregrasp=True,pregrasp_tick=int(env.episode_length_buf[i]))
                        self.grasp_target[i]=self.planner.goal(int(i),False);self.stage_ticks[i]=0;continue
                    desired=path[k]
                delta=desired[:7]-joint[i,:7];fraction=min(1.,.035/max(np.abs(delta).max(),1e-8))
                intermediate=joint[i].copy();intermediate[:7]+=delta*fraction;goal=self.planner.fk(intermediate)
                # Execute the desired checked joint-path displacement through the
                # actual unchanged six-dimensional DLS mapping, never joint targets.
                weight=np.ones(7);weight[4:6]=3
                m=mapping[i]
                path_command=np.linalg.solve(m.T@(weight[:,None]*m)+1e-5*np.eye(6),m.T@(weight*(delta*fraction)))
                path_command/=max(1.,np.max(np.abs(path_command)))
            else:goal=self.grasp_target[i]
            quaternion=pin.Quaternion(goal.rotation).coeffs()
            tp=torch.tensor(goal.translation,device=env.device,dtype=torch.float32)[None]
            tq=torch.tensor([[quaternion[3],*quaternion[:3]]],device=env.device,dtype=torch.float32)
            pe,re=compute_pose_error(p[i:i+1],q[i:i+1],tp,tq,rot_error_type='axis_angle')
            sc=scale[i] if scale.ndim==2 else scale
            action[i,:6]=(torch.cat((pe,re),-1)/sc).clamp(-1,1)[0]
            if phase=='pregrasp':action[i,:6]=torch.tensor(path_command,device=env.device,dtype=torch.float32)
            action[i,3:6].clamp_(-.3,.3)
            if phase in ['align','grasp','probe']:
                if phase=='align' and float(pe.norm())<.010 and float(re.norm())<.12:
                    self.phase[i]='grasp';self.stage_ticks[i]=0
                if self.phase[i] in ['grasp','probe']:
                    action[i,6]=-1
                    valid,measure=valid_grasp(self.planner,int(i),joint[i],gt[i],goal)
                    meta['last_grasp_check']=measure
                    self.grasp_run[i]=self.grasp_run[i]+1 if valid else 0
                    if self.phase[i]=='probe':
                        # Symmetric 1mm probe via original action interface, with closed fingers.
                        direction=self.planner.goal(int(i),True).translation-goal.translation
                        direction/=max(np.linalg.norm(direction),1e-9)
                        offset=direction*.001*(1 if self.stage_ticks[i]<=3 else -1 if self.stage_ticks[i]<=6 else 0)
                        sc=scale[i] if scale.ndim==2 else scale
                        action[i,:3]+=torch.tensor(offset,device=env.device)/sc[:3]
                        action[i,:6].clamp_(-1,1)
                        if not valid:
                            self.phase[i]='grasp';self.stage_ticks[i]=0;meta['probe_losses']=meta.get('probe_losses',0)+1
                        elif self.stage_ticks[i]>=12 and self.grasp_run[i]>=12:
                            self.ready[i]=True;self.phase[i]='ready';meta.update(regrasp=True,valid_grasp=True,probe_survived=True,failure_stage=None)
                    elif self.grasp_run[i]>=6:
                        if self.strict:
                            self.phase[i]='probe';self.stage_ticks[i]=0;meta['contact_hold_ticks']=int(self.grasp_run[i])
                        else:
                            self.ready[i]=True;self.phase[i]='ready';meta.update(regrasp=True,handback=True,
                                handback_tick=int(env.episode_length_buf[i]),failure_stage=None)
        # No joint teleport or joint-target control. Original projection then model checks.
        action=project(env,action,torch.tensor(active,device=env.device))
        for i in np.flatnonzero(active):
            if self.phase[i] in ['failed','ready','release']:continue
            trial=joint[i].copy();trial[:7]+=mapping[i]@action[i,:6].cpu().numpy()
            world=self.planner.world(int(i))
            allowed=self.phase[i] in ['align','grasp','probe']
            accepted=self.planner.escape_edge(joint[i],trial,world) if self.phase[i]=='retreat' else self.planner.edge(joint[i],trial,world,allow_handle=allowed)
            if not accepted:
                action[i,:6]=0;self.current[i]['tracking_collision_rejections']=self.current[i].get('tracking_collision_rejections',0)+1
        return action.cpu().numpy(),time.perf_counter()-wall
