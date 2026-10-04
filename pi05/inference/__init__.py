"""Chunk caches reset on every reset and matched pressure handoff."""
import time
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np,torch,h5py

class PolicyClient(torch.nn.Module):
    def __init__(self,socket,mean,std,seed=143801):
        super().__init__();deadline=time.monotonic()+600
        while True:
            try:self.conn=Client(str(socket),family='AF_UNIX',authkey=b'phase14_3_local');break
            except (FileNotFoundError,ConnectionRefusedError):
                if time.monotonic()>deadline:raise
                time.sleep(.5)
        self.mean=np.asarray(mean,np.float32);self.std=np.asarray(std,np.float32);self.seed=seed;self.ablation=None;self.configure()
    def rpc(self,d):
        self.conn.send(d);r=self.conn.recv()
        if 'error' in r:raise RuntimeError(r['error'])
        return r
    def load(self,path):return self.rpc({'op':'load','checkpoint':str(path)})['meta']
    def configure(self,cohort=None):
        self.cache=np.zeros((32,10,7),np.float32);self.replan=np.full(32,-1000);self.last=np.full(32,-1);self.episode=np.zeros(32,int)
        self.latencies=[];self.clips=[];self.handoffs=[];self.batch=-1
        self.cache_resets=0;self.handoff_plans=0;self.previous_action=None;self.jumps=[];self.replan_jumps=[]
        if cohort:
            with h5py.File(cohort,'r') as h:values=[len(h[k]['prefix_actions'])+1 for k in sorted(h)]
            self.handoffs=[np.array(values[i:i+32],int) for i in range(0,len(values),32)]
        self.is_pressure=bool(cohort)
    def forward(self,x):
        raw=x.detach().cpu().numpy()*self.std+self.mean;age=np.rint(raw[:,25]*600).astype(int)
        if self.ablation=='all_gt':raw[:,26:]=self.mean[26:]
        elif self.ablation=='handle_pose':raw[:,32:]=self.mean[32:]
        if self.is_pressure and np.all(age==0) and (self.batch<0 or np.any(self.last>0)):
            self.batch+=1
            if self.batch>=len(self.handoffs):raise RuntimeError('Unexpected pressure reset')
        reset=age<self.last;self.cache_resets+=int(reset.sum());self.episode+=reset;self.replan[reset]=-1000
        active=np.ones(32,bool)
        if self.is_pressure:
            thresholds=np.full(32,10000)
            if self.batch>=0:thresholds[:len(self.handoffs[self.batch])]=self.handoffs[self.batch]
            active=age>=thresholds
        need=active&((age-self.replan>=5)|reset)
        if np.any(need):
            ids=np.flatnonzero(need);seeds=self.seed+ids*1000003+self.episode[ids]*1009+age[ids]*13
            if self.is_pressure:self.handoff_plans+=int(np.sum(need&(age==thresholds)))
            result=self.rpc({'op':'infer','raw':raw[ids],'noise_seeds':seeds.tolist()})
            self.cache[ids]=result['actions'];self.replan[ids]=age[ids]
            self.latencies.append(result['latency_s']);self.clips.append(result['clip_fraction'])
        cursor=np.clip(age-self.replan,0,4)
        action=self.cache[np.arange(32),cursor].copy();action[~active]=0
        if self.previous_action is not None:
            valid=active&~reset&(self.last>=0)
            jump=np.linalg.norm(action-self.previous_action,axis=-1)
            self.jumps.extend(jump[valid].tolist());self.replan_jumps.extend(jump[valid&need].tolist())
        self.previous_action=action.copy();self.last=age
        return torch.as_tensor(action,device=x.device,dtype=torch.float32)
    def stats(self):
        cumulative=self.rpc({'op':'stats'})
        return {'calls':len(self.latencies),'mean_s':float(np.mean(self.latencies)) if self.latencies else None,
                'p95_s':float(np.quantile(self.latencies,.95)) if self.latencies else None,
                'checkpoint_cumulative':cumulative,'mean_clip_fraction':float(np.mean(self.clips)) if self.clips else 0,'execute':5,'horizon':10,
                'reset_clears':self.cache_resets,'handoff_replans':self.handoff_plans,
                'mean_action_jump_l2':float(np.mean(self.jumps)) if self.jumps else None,
                'mean_replan_jump_l2':float(np.mean(self.replan_jumps)) if self.replan_jumps else None}
    def stop(self):self.rpc({'op':'stop'});self.conn.close()
