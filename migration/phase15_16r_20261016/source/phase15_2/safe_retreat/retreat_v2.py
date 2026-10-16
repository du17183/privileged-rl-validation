"""R0 reference, R1 normal, R2 state-aware, R3 normal then common waypoint.

Every executed proposal uses the original monotone escape check. A rejected
model collision is not a measured physical collision.
"""
import numpy as np
from phase15_1.severe_reacquire.safe_retreat import retreat_command as reference
from .retreat_v1 import normal_target


def retreat_command(planner,i,joint,mapping,scale):
    mode=planner.retreat_mode
    if mode=='R0':return reference(planner,i,joint,mapping,scale)
    world=planner.world(i);before,violation=planner.profile(joint,world)
    pose=planner.fk(joint);target,normal,center=normal_target(planner,i,joint,.08 if mode=='R1' else .05)
    travel=float(np.dot(pose.translation-planner.retreat_origins[i],normal))
    if mode=='R3' and travel>.035:target=planner.fk(planner.clear).translation.copy()
    direction=target-pose.translation;distance=float(np.linalg.norm(direction));direction/=max(distance,1e-8)
    candidates=[]
    directions=[direction]
    if mode in ['R2','R3']:
        handle_goal=planner.goal(i,False).translation
        away=pose.translation-handle_goal;away/=max(np.linalg.norm(away),1e-8)
        directions.extend([normal,away,normal+np.array([0.,0.,.5]),normal+np.array([0.,.5,0.]),normal+np.array([0.,-.5,0.])])
        desired=np.clip(planner.clear[:7]-joint[:7],-.03,.03)
        weights=1+100*(violation>0)
        u=np.linalg.solve(mapping.T@(weights[:,None]*mapping)+1e-5*np.eye(6),mapping.T@(weights*desired))
        candidates.extend([u/max(1.,np.max(np.abs(u)))*factor for factor in [.25,.5,1.]])
        for axis in range(3):
            for sign in [-1,1]:
                r=np.zeros(6);r[axis+3]=sign*.015/scale[axis+3];candidates.append(r)
        # Original gradient/LP only proposes candidates; authoritative checks unchanged.
        old,_,_=reference(planner,i,joint,mapping,scale)
        if old is not None:candidates.append(old)
    for d in directions:
        d=d/max(np.linalg.norm(d),1e-8)
        for step in [.002,.004,.006]:candidates.append(np.r_[d*step,np.zeros(3)]/scale)
    best=None;score=float('inf')
    for u in candidates:
        u=np.clip(u,-1,1);delta=mapping@u
        if np.max(np.abs(delta))>.06:continue
        trial=joint.copy();trial[:7]+=delta
        if not planner.escape_edge(joint,trial,world):continue
        penetration,margin=planner.profile(trial,world)
        value=100*margin.sum()+30*sum(penetration.values())+np.linalg.norm(planner.fk(trial).translation-target)
        if value<score:score=value;best=u
    clear=not before and not np.any(violation>1e-5)
    arrived=distance<.012 if mode in ['R1','R3'] else (travel>.025 and float(np.dot(pose.translation-center,normal))>.04)
    return best,bool(clear and arrived),dict(retreat_mode=mode,target_position=target.tolist(),door_plane_normal=normal.tolist(),model_penetrations=before,margin_violations=violation.tolist(),escape_candidate_found=best is not None,collision_free_model=clear,safe_pose_arrived=bool(arrived),normal_travel_m=travel,predicted_model_only=True)
