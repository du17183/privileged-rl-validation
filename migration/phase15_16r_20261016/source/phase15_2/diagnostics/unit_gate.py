"""Behaviour gates for entry rejection, transition loss, hysteresis and progress."""
import numpy as np
from phase15_2.router.state_machine_v2 import Router,NORMAL,RECOVERY,REACQUIRE,TRANSITION
from phase15_2.router.transition_handoff import interpolate
from phase15_2.common import RESULTS,save_json

def run():
    raw=np.zeros((32,39),np.float32);raw[:,7:9]=.02;raw[:,21]=raw[:,35]=1
    raw[:,18]=raw[:,32]=.3;raw[:,30:32]=1;active=np.zeros(32,bool);active[0]=True
    router=Router(transition=True,progress=True);router.reset(np.arange(32),raw)
    router.state[0]=REACQUIRE;router.entry_allowed[0]=False
    raw[:,25]=200/600
    router.update(raw,active,True,True,np.ones(32,bool))
    assert router.state[0]==REACQUIRE and router.switches[0]==0
    router.entry_allowed[0]=True
    raw[0,32]=.37
    router.update(raw,active,True,True,np.ones(32,bool))
    assert router.state[0]==REACQUIRE and router.switches[0]==0
    raw[0,32]=.3
    router.update(raw,active,True,True,np.ones(32,bool))
    assert router.state[0]==TRANSITION and router.switches[0]==1
    raw[:,25]=214/600;router.update(raw,active,True,True)
    assert router.state[0]==TRANSITION
    raw[:,25]=215/600;router.update(raw,active,True,True)
    assert router.state[0]==NORMAL and router.switches[0]==2
    raw[:,25]=275/600;router.progress_tick[0]=275
    router.update(raw,active,True,True)
    assert router.state[0]==RECOVERY and router.recovery_entries[0]==1
    router.reset(np.arange(32),raw);router.state[0]=TRANSITION;router.entry[0]=275
    raw[0,30:32]=0;raw[:,25]=276/600
    router.update(raw,active,True,True)
    assert router.state[0]==RECOVERY
    assert router.recovery_entries[0]==1
    # A rejected Recovery -> Normal request must not count an accepted switch.
    raw[0,30:32]=1;raw[:,26]=.1;router.reset(np.arange(32),raw)
    router.state[0]=RECOVERY;router.entry_angle[0]=0;router.grasp_run[0]=20
    router.entry[0]=200;raw[:,25]=240/600;router.entry_allowed[0]=False
    router.update(raw,active,True,True)
    assert router.state[0]==RECOVERY and router.switches[0]==0
    normal=np.ones((2,7));recovery=np.zeros((2,7))
    blended=interpolate(normal,recovery,np.array([0,14]))
    assert np.all(blended[0]>0) and np.all(blended[0]<1) and np.all(blended[1]==1)
    save_json(RESULTS/'unit_gate.json',dict(passed=True,checks=['entry_rejection','15tick_transition',
        'post_handoff_progress_guard','transition_contact_loss','accurate_switch_counts','blend_endpoints'],training=False,rl_updates=0))

if __name__=='__main__':run()
