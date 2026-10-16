"""Frozen actor inference with isolated recovery-chain control variants."""
import time
import numpy as np
import torch
from phase15.recovery_policy.state import expand
from phase15.policy_client import ModularPolicy
from phase15_1.policy_client import SystemPolicy
from phase15_2.common import redirect_calibration
from phase15_2.router.state_machine_v2 import Router,NORMAL,RECOVERY,REACQUIRE,TRANSITION,NAMES
from phase15_2.router.transition_handoff import interpolate

class ChainPolicy(SystemPolicy):
    def __init__(self,socket,env,job):
        self.set_job(job);self.writer=None
        super().__init__(socket,env,dict(controller='B',schedule=dict(normal=15,recovery=15),horizons=dict(normal=15,recovery=15)))

    def set_job(self,job):
        old=getattr(self,'planner',None)
        if hasattr(old,'parallel_pool'):old.close()
        self.method=job.get('method','A');self.retreat=job.get('retreat','R0' if self.method=='A' else 'R2')
        self.cpu_geometry_workers=int(job.get('cpu_geometry_workers',0))
        if self.cpu_geometry_workers and (self.method=='A' or self.cpu_geometry_workers not in range(1,5) or job.get('cpu_geometry_version') not in ['parallel_geometry_v1','parallel_geometry_v2']):
            raise RuntimeError('Invalid authorized CPU geometry implementation')
        if self.cpu_geometry_workers:
            import os
            from pathlib import Path
            born=int((Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()[19])
            if born<=job['cpu_geometry_activation_boot_ticks']:
                self.cpu_geometry_workers=0
                print('CPU_GEOMETRY_SERIAL worker started before job activation; preserving serial implementation',flush=True)
        self.chain_config=dict(method=self.method,retreat=self.retreat,geometry=self.method=='E',strict=self.method in ['C','D','E'],transition=self.method in ['D','E'],progress=self.method=='E')
        self.chain_config['entry_distance_exit_m']=.06 if self.chain_config['strict'] else None
        if self.cpu_geometry_workers:
            from phase15_2.common import ROOT,sha256
            for relative,digest in job['cpu_geometry_source_sha256'].items():
                if sha256(ROOT/relative)!=digest:raise RuntimeError('CPU geometry source identity mismatch: '+relative)
            self.chain_config['cpu_geometry']=dict(version=job['cpu_geometry_version'],workers=self.cpu_geometry_workers,
                semantic_gate=job['cpu_geometry_gate'],source_sha256=job['cpu_geometry_source_sha256'],
                numerical_scope='same original kernels; independent CPU processes; original environment order',
                timing_scope='parallel CPU geometry within batch32 evaluation, excludes exclusive batch1 deployment')
            if job['cpu_geometry_version']=='parallel_geometry_v2':
                self.chain_config['cpu_geometry'].update(retreat_execution='original_serial',
                    parallel_operations=['tracking','normal_candidates'])
        self.variant='B';self.schedule=dict(normal=15,recovery=15);self.horizons=dict(normal=15,recovery=15)
        self.force_planner=job.get('force_planner',False);self.capture_trace=False;self.planner=None

    def load(self,checkpoints,arm):
        redirect_calibration()
        result=super().load(checkpoints,arm)
        if self.method!='A':
            if self.cpu_geometry_workers:
                from phase15_2.parallel_recovery import Reacquire
                self.planner=Reacquire(self.env,'B',retreat=self.retreat,geometry=self.method=='E',strict=self.method in ['C','D','E'],workers=self.cpu_geometry_workers)
            else:
                from phase15_2.recovery_chain import Reacquire
                self.planner=Reacquire(self.env,'B',retreat=self.retreat,geometry=self.method=='E',strict=self.method in ['C','D','E'])
        return result

    def stop(self):
        try:return super().stop()
        finally:
            if hasattr(self.planner,'parallel_pool'):self.planner.close()

    def configure(self,cohort=None):
        super().configure(cohort)
        self.phase_ticks=np.zeros((32,4),int)
        if self.method in ['C','D','E']:self.router=Router(transition=self.method in ['D','E'],progress=self.method=='E')
        self.candidate_checked=np.full(32,-1000,int);self.candidate_safe=np.zeros(32,bool)
        self.entry_checks=[None]*32;self.normal_entries=[[] for _ in range(32)]
        self.stage_seen=np.full((32,9),-1,int);self.stage_progress_run=np.zeros(32,int)
        self.probe_start=np.full(32,-1,int);self.probe_done=np.zeros(32,bool);self.probe_origin=np.zeros((32,3))
        self.trans_cache={m:np.zeros((32,15,7),np.float32) for m in ['normal','recovery']}
        self.trans_replan=np.full(32,-1000,int);self.entry_latency=[];self.transition_latency=[]
        self.router_latency=[];self.planner_latency=[]

    def _reset(self,ids,raw):
        super()._reset(ids,raw)
        if hasattr(self,'candidate_checked'):
            self.candidate_checked[ids]=-1000;self.candidate_safe[ids]=False;self.trans_replan[ids]=-1000
            self.stage_seen[ids]=-1;self.stage_progress_run[ids]=0
            self.probe_start[ids]=-1;self.probe_done[ids]=False
            for i in ids:self.entry_checks[i]=None;self.normal_entries[i]=[]

    def begin_trace(self,path):
        self.close_trace()
        from phase15_2.evaluation.stage_eval import StageWriter
        self.writer=StageWriter(path.with_suffix('.stage_trace.h5'))

    def close_trace(self):
        if getattr(self,'writer',None):self.writer.close();self.writer=None

    def forward(self,x):
        tick=time.perf_counter()
        # Exact Phase15.1 E_final baseline action path.
        action=SystemPolicy._forward(self,x) if self.method=='A' else self._forward(x)
        self.control_latencies.append(time.perf_counter()-tick);self.before_action=self.current_raw.copy()
        self.before_tool=self.env.get_tool_state().cpu().numpy().copy()
        for i in np.flatnonzero(self.active):
            self.phase_ticks[i,self.router.state[i]]+=1
            e=self.event[i]
            if e is not None and self.stage_seen[i,0]<0:self.stage_seen[i,0]=e['tick']
            if self.planner is not None:
                meta=self.planner.current[i]
                if meta:
                    for k,field in [(1,'safe_retreat'),(2,'reached_pregrasp'),(5,'regrasp'),(6,'handback')]:
                        if meta.get(field) and self.stage_seen[i,k]<0:self.stage_seen[i,k]=round(self.current_raw[i,25]*600)
        return action

    def _mapping(self):
        term=self.env.action_manager.get_term('arm_action');jac=term._compute_frame_jacobian().clone();damping=term._ik_controller.cfg.ik_params['lambda_val']
        inverse=torch.linalg.solve(jac@jac.transpose(1,2)+damping**2*torch.eye(6,device=self.env.device),jac).transpose(1,2)
        scale=term._scale
        return (inverse*scale[:,None,:] if scale.ndim==2 else inverse*scale).cpu().numpy()

    def safe_candidates(self,raw,ids,actions):
        mapping=self._mapping();checked=np.zeros(32,bool)
        if hasattr(self.planner,'parallel_pool'):
            return self.planner.safe_candidates(raw,ids,actions,mapping)
        for j,i in enumerate(ids):
            q=raw[i,:9].copy();world=self.planner.planner.world(int(i));ok=True
            for action in actions[j,:3]:
                if action[6]>=0:ok=False;break
                trial=q.copy();trial[:7]+=mapping[i]@action[:6]
                if not self.planner.planner.edge(q,trial,world,allow_handle=True):ok=False;break
                q=trial
            checked[i]=ok
        return checked

    def prepare_entry(self,raw,active,age):
        from phase15_2.grasp_validation.valid_grasp import normal_entry
        pending=active&((self.router.state==RECOVERY)|(self.router.state==TRANSITION)|((self.router.state==REACQUIRE)&self.planner.ready))
        allowed=np.zeros(32,bool)
        for i in np.flatnonzero(pending):
            allowed[i],self.entry_checks[i]=normal_entry(self.planner.planner,int(i),raw[i])
        ids=np.flatnonzero(allowed&(age-self.candidate_checked>=15))
        if len(ids):
            wall=time.perf_counter();seeds=self.seed+ids*1000003+self.episode[ids]*1009+age[ids]*13
            answer=self.rpc(dict(op='infer',module='normal',raw=raw[ids],noise_seeds=seeds.tolist(),horizon=15))
            safe=self.safe_candidates(raw,ids,answer['actions']);self.candidate_safe[ids]=safe[ids];self.candidate_checked[ids]=age[ids]
            self.entry_latency.append(time.perf_counter()-wall)
            self.requests.append(dict(module='normal_entry_check',batch=len(ids),latency_s=answer['latency_s'],service_s=answer['service_s'],rpc_s=self.entry_latency[-1],execute_ticks=0,horizon=15,role='candidate_only'))
        self.entry_geometry_allowed=allowed
        recovery=active&(self.router.state==RECOVERY)
        self.probe_start[recovery&~allowed]=-1;self.probe_done[recovery&~allowed]=False
        begin=recovery&allowed&(self.probe_start<0)
        self.probe_start[begin]=age[begin];self.probe_origin[begin]=raw[begin,18:21]
        self.probe_done|=recovery&allowed&(self.probe_start>=0)&(age-self.probe_start>=12)
        self.router.entry_allowed=allowed&self.candidate_safe&(age-self.candidate_checked<15)&(~recovery|self.probe_done)

    def transition_action(self,raw,active,age,changed,action):
        ids=np.flatnonzero(active&(self.router.state==TRANSITION))
        if not len(ids):return
        wall=time.perf_counter();need=ids[(age[ids]-self.trans_replan[ids]>=15)|changed[ids]]
        if len(need):
            for module in ['normal','recovery']:
                value=raw[need] if module=='normal' else expand(raw[need]);seeds=self.seed+need*1000003+self.episode[need]*1009+age[need]*13
                started=time.perf_counter();answer=self.rpc(dict(op='infer',module=module,raw=value,noise_seeds=seeds.tolist(),horizon=15));elapsed=time.perf_counter()-started
                self.trans_cache[module][need]=answer['actions']
                for i in need:self.calls[i][module]+=1;self.costs[i][module]+=elapsed
                self.requests.append(dict(module=module,batch=len(need),latency_s=answer['latency_s'],service_s=answer['service_s'],rpc_s=elapsed,clip_fraction=answer['clip_fraction'],execute_ticks=15,horizon=15,role='transition'))
            self.trans_replan[need]=age[need]
        cursor=np.clip(age[ids]-self.trans_replan[ids],0,14)
        blend=interpolate(self.trans_cache['normal'][ids,cursor],self.trans_cache['recovery'][ids,cursor],age[ids]-self.router.entry[ids])
        safe=self.safe_candidates(raw,ids,blend[:,None,:])
        action[ids]=blend
        rejected=ids[~safe[ids]];action[rejected]=0;action[rejected,6]=-1;self.router.transition_rejected[rejected]=True
        self.transition_latency.append(time.perf_counter()-wall)

    def observe(self,robot,gt,done):
        r=robot.cpu().numpy();g=gt.cpu().numpy();d=done.cpu().numpy();old=len(self.telemetry)
        for i in np.flatnonzero(self.active):
            age=round(r[i,25]*600);distance=np.linalg.norm(r[i,18:21]-g[i,2:5]);contact=bool(np.all(g[i,9:11]>.5))
            if self.stage_seen[i,0]>=0:
                for k,ok in [(3,distance<.035),(4,contact),(8,d[i] and g[i,0]>1.)]:
                    if ok and self.stage_seen[i,k]<0:self.stage_seen[i,k]=age
                e=self.event[i]
                if e and e.get('returned_normal'):
                    if self.stage_seen[i,6]<0:self.stage_seen[i,6]=e['return_tick']
                    if not self.normal_entries[i] or self.normal_entries[i][-1]['tick']!=e['return_tick']:
                        self.normal_entries[i].append(dict(tick=e['return_tick'],door_angle=float(g[i,0]),entry_checks=self.entry_checks[i]))
                        self.stage_progress_run[i]=0
                    mark=self.normal_entries[i][-1]['door_angle']
                    self.stage_progress_run[i]=self.stage_progress_run[i]+1 if g[i,0]>mark+.03 else 0
                    if self.stage_progress_run[i]>=6 and self.stage_seen[i,7]<0:self.stage_seen[i,7]=age
        if self.writer is not None:self.writer.observe(self,robot,gt,done)
        ModularPolicy.observe(self,robot,gt,done)
        for row in self.telemetry[old:]:
            i=row['clone'];row['phase_ticks']={name:int(self.phase_ticks[i,j]) for j,name in enumerate(NAMES)}
            row['chain_stage_first_tick']=self.stage_seen[i].tolist();row['normal_entries']=self.normal_entries[i]
            row['last_entry_check']=self.entry_checks[i];row['chain_config']=self.chain_config

    def stats(self):
        result=SystemPolicy.stats(self);result['chain_config']=self.chain_config
        for name,values in [('entry_check',self.entry_latency),('transition',self.transition_latency),('control',self.control_latencies),('router',self.router_latency),('planner',self.planner_latency)]:
            result[name+'_latency']={f'p{p}_s':float(np.quantile(values,p/100)) if values else None for p in [50,95,99]}
        result['stage_trace_sampling']='every10ticks or stage/state change or terminal'
        return result

    def _forward(self, x):
        raw = x.detach().cpu().numpy() * self.std + self.mean
        age = np.rint(raw[:, 25] * 600).astype(int)
        if self.pressure and np.all(age == 0) and (self.batch < 0 or np.any(self.last > 0)):
            self.batch += 1
            if self.batch >= len(self.thresholds):
                raise RuntimeError('Unexpected pressure batch reset')
            self.finished[:] = False
            self.handed[:] = False
        reset = (age < self.last) | (self.last < 0)
        self.episode += (age < self.last)
        self.reset_clears += int(reset.sum())
        self.replan[reset] = -1000
        self._reset(np.flatnonzero(reset), raw)
        active = np.ones(32, bool)
        handed = np.zeros(32, bool)
        thresholds = np.zeros(32, int)
        if self.pressure:
            thresholds[:] = 10000
            if self.batch >= 0:
                thresholds[:len(self.thresholds[self.batch])] = self.thresholds[self.batch]
            # The historical pressure evaluator keeps stepping auto-reset
            # clones while other samples finish. They are no longer scored.
            # Do not route, infer or count a second takeover for those clones.
            active = (age >= thresholds) & ~self.finished
            # Never switch/infer on prerecorded actions which are overwritten.
            handed = active & ~self.handed
            self.handed[handed] = True
            self.handoff_replans += int(handed.sum())
            self.replan[handed] = -1000
            self.cache[handed] = 0
            history = [v.copy() for v in [self.router.previous_contact, self.router.peak,
                       self.router.progress_mark, self.router.progress_tick]]
            self._reset(np.flatnonzero(handed), raw)
            for array, old in zip([self.router.previous_contact, self.router.peak,
                                  self.router.progress_mark, self.router.progress_tick], history):
                array[handed] = old[handed]
            prefix = ~active & ~self.finished & (thresholds < 10000)
            contact = np.all(raw[:, 30:32] > .5, axis=-1) & (raw[:, 26] > .05)
            self.router.previous_contact |= prefix & contact
            self.router.peak[prefix] = np.maximum(self.router.peak[prefix], raw[prefix, 26])
            advanced = prefix & (raw[:, 26] > self.router.progress_mark + .015)
            self.router.progress_mark[advanced] = raw[advanced, 26]
            self.router.progress_tick[advanced] = age[advanced]
        tick = time.perf_counter()
        ready = self.planner.ready.copy() if self.planner is not None else None
        if self.method in ['C','D','E']:
            self.prepare_entry(raw,active,age)
        before, changed, reasons = self.router.update(raw, active, self.arm in ['C', 'D'],
                                                       self.arm == 'D', ready)
        if self.force_planner and np.any(handed):
            self.router.state[handed] = REACQUIRE
            changed = self.router.state != before
            reasons[handed] = 'graded_direct_reacquire'
            self.router.entry[handed] = age[handed]
            self.router.planner_entries[handed] += 1
        router_dt = time.perf_counter() - tick
        self.router_latency.append(router_dt)
        for i in np.flatnonzero(active):
            self.costs[i]['router'] += router_dt
            new_failure = changed[i] and self.router.state[i] != NORMAL and (
                self.event[i] is None or self.event[i].get('returned_normal', False))
            if reasons[i] and (self.event[i] is None or new_failure):
                event = dict(tick=int(age[i]), trigger=str(reasons[i]),
                    detection=True, switch_correct=bool(self.arm in ['C', 'D']),
                    route=NAMES[int(self.router.state[i])], initial_angle=float(raw[i, 26]),
                    initial_distance=float(np.linalg.norm(raw[i, 32:35] - raw[i, 18:21])),
                    flags=[False] * 4, runs=[0] * 4, returned_normal=False)
                self.events[i].append(event)
                self.event[i] = event
        self.replan[changed] = -1000
        self.cache[changed] = 0
        self.switch_clears += int(changed.sum())
        for i in np.flatnonzero(changed & (self.router.state == NORMAL)):
            if self.event[i] is not None:
                self.event[i]['returned_normal'] = True
                self.event[i]['return_tick'] = int(age[i])
            if self.planner is not None and self.planner.current[i] is not None:
                self.planner.current[i].update(handback=True,handback_tick=int(age[i]))
            # Keep stage tracking until episode terminal; do not invent another
            # failure event from the same resolved contact-loss interval.
        enters = np.flatnonzero(changed & (self.router.state == REACQUIRE))
        if len(enters):
            self.planner.start(enters)
        action = np.zeros((32, 7), np.float32)
        planning = active & (self.router.state == REACQUIRE)
        if np.any(planning):
            planner_action, timing = self.planner.action(planning)
            self.planner_latency.append(timing)
            action[planning] = planner_action[planning]
            for i in np.flatnonzero(planning):
                self.costs[i]['planner'] += timing
        for state, module in [(NORMAL, 'normal'), (RECOVERY, 'recovery')]:
            eligible = active & (self.router.state == state)
            need = eligible & ((age - self.replan >= self.schedule[module]) | reset | changed)
            if np.any(need):
                ids = np.flatnonzero(need)
                seeds = self.seed + ids * 1000003 + self.episode[ids] * 1009 + age[ids] * 13
                values = raw[ids] if module == 'normal' else expand(raw[ids])
                wall = time.perf_counter()
                result = self.rpc(dict(op='infer', module=module, raw=values, noise_seeds=seeds.tolist(), horizon=self.horizons[module]))
                wall_dt = time.perf_counter() - wall
                self.cache[ids] = 0
                self.cache[ids, :self.horizons[module]] = result['actions']
                self.replan[ids] = age[ids]
                self.requests.append(dict(module=module, batch=len(ids), latency_s=result['latency_s'],
                    service_s=result['service_s'], rpc_s=wall_dt, clip_fraction=result['clip_fraction'],
                    execute_ticks=self.schedule[module], horizon=self.horizons[module], age=int(age[ids[0]])))
                for i in ids:
                    self.costs[i][module] += wall_dt
                    self.calls[i][module] += 1
            ids = np.flatnonzero(eligible)
            cursor = np.clip(age[ids] - self.replan[ids], 0, self.schedule[module]-1)
            action[ids] = self.cache[ids, cursor]
        if self.method in ['C','D','E']:
            probe=active&(self.router.state==RECOVERY)&(self.probe_start>=0)&~self.probe_done
            for i in np.flatnonzero(probe):
                elapsed=age[i]-self.probe_start[i]
                direction=self.planner.planner.goal(int(i),True).translation-self.planner.planner.goal(int(i),False).translation
                direction/=max(np.linalg.norm(direction),1e-9)
                target=self.probe_origin[i]+direction*.001*(1 if elapsed<3 else -1 if elapsed<6 else 0)
                u=np.zeros(7,np.float32);u[:3]=np.clip((target-raw[i,18:21])/.05,-1,1);u[6]=-1
                if self.safe_candidates(raw,np.array([i]),u[None,None,:])[i]:action[i]=u
                else:
                    action[i]=0;action[i,6]=-1;self.probe_start[i]=-1;self.probe_done[i]=False
            self.stage_seen[active&self.probe_done&(self.stage_seen[:,5]<0),5]=age[active&self.probe_done&(self.stage_seen[:,5]<0)]
        if self.method in ["D","E"]:
            self.transition_action(raw,active,age,changed,action)
        for i in np.flatnonzero(handed):
            index = self.batch * 32 + int(i)
            planner = self.router.state[i] == REACQUIRE
            refreshed = planner or self.replan[i] == age[i]
            self.handoff_records.append(dict(cohort_id=self.cohort_ids[index],
                tick=int(age[i]), cache_cleared=True, refreshed=bool(refreshed),
                module=NAMES[int(self.router.state[i])]))
            if not refreshed:
                raise RuntimeError('Action cache was not refreshed at physical takeover')
        self.active = active
        self.current_raw = raw
        self.last_action = action
        self.last = age
        return torch.as_tensor(action, dtype=torch.float32, device=x.device)

