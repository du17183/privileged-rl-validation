"""Four-state routing with checked entry and optional resumed-progress guard."""
import numpy as np
from phase15.router.recovery_state_machine import Router as Original
NORMAL,RECOVERY,REACQUIRE,TRANSITION=range(4)
NAMES=['normal','recovery','reacquire','transition']


class Router(Original):
    def __init__(self,transition=False,progress=False):
        super().__init__();self.transition=transition;self.progress=progress
        self.entry_allowed=np.zeros(32,bool);self.transition_rejected=np.zeros(32,bool)
        self.handoff_age=np.full(32,-1,int);self.handoff_angle=np.zeros(32)
        self.progress_run=np.zeros(32,int);self.progress_resumed=np.zeros(32,bool)
        self.previous_angle=np.zeros(32)

    def reset(self,ids,raw):
        super().reset(ids,raw)
        self.entry_allowed[ids]=False;self.transition_rejected[ids]=False
        self.handoff_age[ids]=-1;self.progress_run[ids]=0;self.progress_resumed[ids]=False
        self.handoff_angle[ids]=raw[ids,26];self.previous_angle[ids]=raw[ids,26]

    def update(self,raw,active,modular,planner_enabled,planner_ready=None):
        before=self.state.copy();age=np.rint(raw[:,25]*600).astype(int)
        prior_recovery_entries=self.recovery_entries.copy();prior_planner_entries=self.planner_entries.copy()
        names=['entry','last_switch','peak','progress_mark','progress_tick','contact_lost','switches','entry_distance','entry_angle']
        saved={k:getattr(self,k).copy() for k in names}
        _,_,reasons=super().update(raw,active,modular,planner_enabled,None)
        legacy_return=(before==RECOVERY)&(self.state==NORMAL)
        self.state[legacy_return]=RECOVERY
        for k in names:getattr(self,k)[legacy_return]=saved[k][legacy_return]
        ready=np.zeros(32,bool) if planner_ready is None else planner_ready
        recovery_ready=legacy_return&(self.grasp_run>=12)
        distance=np.linalg.norm(raw[:,32:35]-raw[:,18:21],axis=1)
        allow=active&self.entry_allowed&(distance<.06)&(ready&(before==REACQUIRE)|recovery_ready)
        self.state[allow]=TRANSITION if self.transition else NORMAL
        transitioning=active&(before==TRANSITION)
        distance=np.linalg.norm(raw[:,32:35]-raw[:,18:21],axis=1)
        contact=np.all(raw[:,30:32]>.5,axis=1)
        bad=transitioning&((~contact)|(distance>.08)|self.transition_rejected)
        self.state[bad]=RECOVERY;self.state[bad&(distance>.12)]=REACQUIRE
        reasons[bad]='transition_contact_or_safety_failure'
        done=transitioning&~bad&(age-self.entry>=15)&self.entry_allowed&(distance<.06)
        self.state[done]=NORMAL
        returned=active&(self.state==NORMAL)&(before!=NORMAL)
        self.handoff_age[returned]=age[returned];self.handoff_angle[returned]=raw[returned,26]
        self.progress_run[returned]=0;self.progress_resumed[returned]=False
        monitor=active&(self.handoff_age>=0)&(self.state==NORMAL)
        advancing=monitor&(raw[:,26]>self.handoff_angle+.03)&(raw[:,26]>=self.previous_angle-.001)
        self.progress_run=np.where(advancing,self.progress_run+1,0)
        self.progress_resumed|=monitor&(self.progress_run>=6)
        if self.progress:
            stalled=monitor&~self.progress_resumed&(age-self.handoff_age>=60)
            self.state[stalled]=RECOVERY;reasons[stalled]='post_handoff_progress_not_resumed'
        self.previous_angle[active]=raw[active,26]
        changed=self.state!=before
        # Recompute switches against pre-update state, avoiding rejected-entry counts.
        self.switches=saved['switches']+changed.astype(int)
        self.entry[changed]=age[changed];self.last_switch[changed]=age[changed]
        self.entry_distance[changed]=distance[changed];self.entry_angle[changed]=raw[changed,26]
        self.recovery_entries=prior_recovery_entries+(changed&(self.state==RECOVERY)).astype(int)
        self.planner_entries=prior_planner_entries+(changed&(self.state==REACQUIRE)).astype(int)
        self.progress_tick[returned]=age[returned];self.progress_mark[returned]=raw[returned,26]
        self.peak[returned]=raw[returned,26];self.contact_lost[returned]=0
        self.transition_rejected[changed]=False
        return before,changed,reasons

    def specification(self):
        return dict(super().specification(),states=NAMES,entry_contact_hold_ticks=12,transition_ticks=15 if self.transition else 0,entry_distance_exit_m=.06,progress_guard=self.progress,progress_gain_rad=.03,progress_consecutive_ticks=6,progress_timeout_ticks=60)
