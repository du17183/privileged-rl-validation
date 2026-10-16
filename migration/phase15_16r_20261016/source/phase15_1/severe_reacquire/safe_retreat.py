"""Bounded release/escape using the original Cartesian-to-DLS action mapping."""
import numpy as np
from scipy.optimize import linprog


def retreat_command(planner,i,joint,mapping,scale):
    world=planner.world(i);before,violation=planner.profile(joint,world)
    desired=np.clip((planner.clear[:7]-joint[:7])*.08,-.04,.04)
    weights=1+80*(violation>0)
    h=mapping.T@(weights[:,None]*mapping)+1e-5*np.eye(6)
    u=np.linalg.solve(h,mapping.T@(weights*desired))
    u/=max(1.,np.max(np.abs(u)))
    pose=planner.fk(joint)
    goal=planner.goal(i,True)
    away=goal.translation-pose.translation;away/=max(np.linalg.norm(away),1e-6)
    candidates=[u*.5,u*.25,u]
    for direction in [away,np.array([-1.,0.,.5]),np.array([0.,0.,1.]),np.array([0.,1.,.3]),np.array([0.,-1.,.3])]:
        candidate=np.r_[direction/np.linalg.norm(direction)*.004,np.zeros(3)]/scale
        candidates.append(candidate)
    if before or np.any(violation>1e-5):
        # Model-distance gradients propose escape commands; full discrete model
        # checks below remain authoritative. No collision rule is relaxed.
        pairs=list(before);derivative=np.zeros((len(pairs),7));epsilon=2e-4
        for k in range(7):
            plus=joint.copy();minus=joint.copy();plus[k]+=epsilon;minus[k]-=epsilon
            dp,_=planner.profile(plus,world);dm,_=planner.profile(minus,world)
            derivative[:,k]=[(dp.get(key,0)-dm.get(key,0))/(2*epsilon) for key in pairs]
        gradient=derivative@mapping
        objective=20*gradient.sum(axis=0)
        signed=np.where(joint[:7]<planner.lower[:7],-1.,np.where(joint[:7]>planner.upper[:7],1.,0.))
        objective+=100*signed@mapping
        constraints=[mapping,-mapping]
        limits=[np.maximum(planner.upper[:7]-joint[:7],0),np.maximum(joint[:7]-planner.lower[:7],0)]
        if len(pairs):constraints.append(gradient);limits.append(np.zeros(len(pairs)))
        solve=linprog(objective,A_ub=np.concatenate(constraints),b_ub=np.concatenate(limits),bounds=[(-.3,.3)]*6,method='highs')
        if solve.success:
            candidates.extend([solve.x*factor for factor in [1.,.5,.25,.1]])
    chosen=None;score=float('inf')
    for u in candidates:
        u=np.clip(u,-1,1);delta=mapping@u
        if np.max(np.abs(delta))>.06:continue
        trial=joint.copy();trial[:7]+=delta
        if not planner.escape_edge(joint,trial,world):continue
        penetration,margin=planner.profile(trial,world)
        value=100*float(margin.sum())+20*sum(penetration.values())+np.linalg.norm(planner.fk(trial).translation-goal.translation)
        # Current penetrations may only decrease; newly introduced pairs rejected.
        if value<score:score=value;chosen=u
    safe=not before and not np.any(violation>1e-5)
    return chosen,safe,dict(model_penetrations=before,margin_violations=violation.tolist(),
        escape_candidate_found=chosen is not None,predicted_model_only=True)
