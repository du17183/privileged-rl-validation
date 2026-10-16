"""Explicit copy of frozen Phase15 execution, with only named controller/schedule changes."""
import time
import numpy as np
import torch
from phase15.recovery_policy.state import expand
from phase15.router.recovery_state_machine import NORMAL,RECOVERY,REACQUIRE,NAMES
from phase15_1.severe_reacquire.failure_analysis import DiagnosticPolicy
from phase15_1.inference_opt.replanning_schedule import HysteresisRouter

class SystemPolicy(DiagnosticPolicy):
    def __init__(self,socket,env,job):
        self.variant=job.get('controller','A')
        self.schedule=job.get('schedule',{'normal':5,'recovery':5})
        self.horizons=job.get('horizons',{'normal':10,'recovery':10})
        self.force_planner=job.get('force_planner',False)
        self.capture_trace=job.get('capture_trace',False)
        super().__init__(socket,env)

    def load(self,checkpoints,arm):
        if self.variant=='A':return super().load(checkpoints,arm)
        result=super().load(checkpoints,'C');self.arm=arm
        from phase15_1.severe_reacquire.reacquire_state_machine import Reacquire
        if self.planner is None:self.planner=Reacquire(self.env,self.variant)
        return result

    def configure(self,cohort=None):
        super().configure(cohort)
        self.control_latencies=[]
        self.phase_ticks=np.zeros((32,3),int)
        self.engineering={}
        if self.variant=='D':self.router=HysteresisRouter()
        self.cache=np.zeros((32,max(self.horizons.values()),7),np.float32)
        if any(self.schedule[k]>self.horizons[k] for k in self.schedule):raise ValueError('No repeated action padding')

    def forward(self,x):
        tick=time.perf_counter();action=self._forward(x)
        self.control_latencies.append(time.perf_counter()-tick)
        self.before_action=self.current_raw.copy()
        for i in np.flatnonzero(self.active):self.phase_ticks[i,self.router.state[i]]+=1
        return action

    def _reset(self,ids,raw):
        super()._reset(ids,raw)
        if hasattr(self,'phase_ticks'):self.phase_ticks[ids]=0

    def rpc(self,request):
        result=super().rpc(request)
        if 'engineering' in result:
            if not hasattr(self,'engineering'):self.engineering={}
            self.engineering[request['module']]=result['engineering']
        return result

    def observe(self,robot,gt,done):
        before=len(self.telemetry)
        if self.capture_trace:super().observe(robot,gt,done)
        else:
            from phase15.policy_client import ModularPolicy
            ModularPolicy.observe(self,robot,gt,done)
        for row in self.telemetry[before:]:
            row['phase_ticks']={name:int(self.phase_ticks[row['clone'],j]) for j,name in enumerate(NAMES)}

    def benchmark(self):
        result={};raw=self.current_raw[:1]
        for module in self.metas:
            value=raw if module=='normal' else expand(raw);times=[]
            for j in range(8):
                wall=time.perf_counter()
                response=self.rpc(dict(op='infer',module=module,raw=value,noise_seeds=[151099+j],horizon=self.horizons[module]))
                if j>=3:times.append(dict(model_s=response['latency_s'],service_s=response['service_s'],rpc_s=time.perf_counter()-wall))
            result[module]=dict(batch=1,horizon=self.horizons[module],execute_ticks=self.schedule[module],samples=times,
                mean_model_s=float(np.mean([v['model_s'] for v in times])),mean_rpc_s=float(np.mean([v['rpc_s'] for v in times])),
                replan_budget_s=self.schedule[module]/60,budget_met=all(v['rpc_s']<=self.schedule[module]/60 for v in times))
        return result

    def stats(self):
        result=super().stats()
        result.update(controller=self.variant,schedule=self.schedule,horizons=self.horizons,
            horizon_extrapolation=any(v>10 for v in self.horizons.values()),
            simulated_replan_hz={k:60/v for k,v in self.schedule.items()},
            e2e_control_mean_s=float(np.mean(self.control_latencies)) if self.control_latencies else None,
            e2e_control_p95_s=float(np.quantile(self.control_latencies,.95)) if self.control_latencies else None,
            deadline_misses={k:sum(v['rpc_s']>self.schedule[k]/60 for v in self.requests if v['module']==k) for k in self.schedule},
            total_simulated_episode_seconds=sum(v['original_episode_ticks']/60 for v in self.telemetry),
            calls_per_episode=float(np.mean([sum(v['pi_calls'].values()) for v in self.telemetry])) if self.telemetry else None,
            physics_waits_for_inference=True)
        result['chunk_length']=self.horizons;result['execute_steps']=self.schedule
        result['engineering']=self.engineering
        result['effective_replan_period_s']={}
        for j,module in enumerate(['normal','recovery']):
            count=sum(r['pi_calls'][module] for r in self.telemetry)
            duration=sum(r.get('phase_ticks',{}).get(module,0) for r in self.telemetry)/60
            result['effective_replan_period_s'][module]=duration/count if count else None
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
        before, changed, reasons = self.router.update(raw, active, self.arm in ['C', 'D'],
                                                       self.arm == 'D', ready)
        if self.force_planner and np.any(handed):
            self.router.state[handed] = REACQUIRE
            changed = self.router.state != before
            reasons[handed] = 'graded_direct_reacquire'
            self.router.entry[handed] = age[handed]
            self.router.planner_entries[handed] += 1
        router_dt = time.perf_counter() - tick
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
            # Keep stage tracking until episode terminal; do not invent another
            # failure event from the same resolved contact-loss interval.
        enters = np.flatnonzero(changed & (self.router.state == REACQUIRE))
        if len(enters):
            self.planner.start(enters)
        action = np.zeros((32, 7), np.float32)
        planning = active & (self.router.state == REACQUIRE)
        if np.any(planning):
            planner_action, timing = self.planner.action(planning)
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

