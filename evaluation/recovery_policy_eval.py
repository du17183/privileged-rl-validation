"""Matched physical action-history replay, then learned policy only.

No recovery expert is imported or called. Preserve original 600-tick budget.
The only physical injection is the prerecorded door regression when requested.
"""
import csv,json,math,types
from pathlib import Path
from collections import defaultdict
import h5py,numpy as np,torch
from door_env.door import robot_observation,door_index
from door_env.isaac_env import transition_after_step
from experiments.phase14_2_recovery_bc.model import measured,inputs

def pressure(env,net,mean,std,meta,cohort,path):
    samples=[]
    with h5py.File(cohort,'r') as h:
        for key in sorted(h):
            g=h[key];samples.append(dict(id=key,clone=int(g.attrs['clone']),actions=g['prefix_actions'][:],
               params=g['reset_parameters'][:],robot=g['handoff_robot'][:],gt=g['handoff_state'][:],
               write_tick=int(g.attrs['door_write_tick']),assignment=int(g.attrs['assignment'])))
    # Same32-clone layout, packed legal resets/actions in workspace coordinates.
    # Validate actual handoff pose after physical replay; no state teleport.
    byclone=defaultdict(list)
    for j,s in enumerate(samples):byclone[j%32].append(s)
    rows=[];prefix_errors=[];interactions=0;initialized={}
    original_sample=env._sample
    try:
        for batch in range(max(map(len,byclone.values()))):
            selected={i:v[batch] for i,v in byclone.items() if len(v)>batch}
            params=np.zeros((32,5),np.float32);params[:,4]=1
            for i,s in selected.items():params[i]=s['params']
            fixed=torch.tensor(params,device=env.device)
            env._sample=types.MethodType(lambda self,ids:fixed[ids].clone(),env)
            env.reset(seed=142801)
            prefix=torch.zeros((32,601,7),device=env.device)
            lens=torch.zeros(32,device=env.device,dtype=torch.long)
            ends={};writes={}
            for i,s in selected.items():
                n=len(s['actions']);prefix[i,:n]=torch.as_tensor(s['actions'],device=env.device);lens[i]=n
                ends.setdefault(n,[]).append(i)
                if s['write_tick']>=0:writes.setdefault(s['write_tick'],[]).append(i)
            active=torch.zeros(32,device=env.device,dtype=torch.bool);complete=set()
            runs=torch.zeros((32,4),device=env.device,dtype=torch.long);flags=torch.zeros((32,4),device=env.device,dtype=torch.bool)
            start=torch.zeros(32,device=env.device);maxangle=torch.zeros_like(start);length=torch.zeros_like(start,dtype=torch.long)
            initial={};begun=set()
            pending=set()
            for tick in range(605):
                robot=robot_observation(env);gt=env.get_tool_state()
                with torch.no_grad():action=net(inputs(robot,measured(env),mean,std,meta['mode']))
                if tick in writes:
                    ids=torch.tensor(writes[tick],device=env.device);cab=env.scene['cabinet'];j=door_index(env)
                    q=cab.data.joint_pos[ids].clone();v=cab.data.joint_vel[ids].clone();q[:,j]=math.radians(20);v[:,j]=0
                    cab.write_joint_state_to_sim(q,v,env_ids=ids)
                    gt=env.get_tool_state()
                if tick<601:action=torch.where((lens>tick)[:,None],prefix[:,tick],action)
                for i in ends.get(tick,[]):
                    s=selected[i]
                    if i not in begun:
                        # Test initialization only: reconstruct this fresh recorded
                        # physical state. Action replay alone drifts under PhysX.
                        # No reconstruction is performed after policy takeover.
                        ids=torch.tensor([i],device=env.device);rob=env.scene['robot'];cab=env.scene['cabinet'];j=door_index(env)
                        rq=torch.as_tensor(s['robot'][:9],device=env.device)[None];rv=torch.as_tensor(s['robot'][9:18],device=env.device)[None]
                        rob.write_joint_state_to_sim(rq,rv,env_ids=ids)
                        cq=cab.data.joint_pos[ids].clone();cv=cab.data.joint_vel[ids].clone();cq[:,j]=float(s['gt'][0]);cv[:,j]=float(s['gt'][1])
                        cab.write_joint_state_to_sim(cq,cv,env_ids=ids)
                        env.sim.forward();env.scene['ee_frame'].reset(ids);env.scene['ee_frame'].update(0.,force_recompute=True)
                        action[i,:6]=0;action[i,6]=float(s['actions'][-1,6]) if len(s['actions']) else 1
                        pending.add(i);begun.add(i)
                policy_active=active.clone()
                _,_,term,trunc,_=env.step(action);nr,ng,done=transition_after_step(env,term,trunc)
                # One neutral physics tick rebuilds real contact forces. It is
                # charged to the original episode budget, never added for free.
                for i in list(pending):
                        s=selected[i]
                        # Natural failure snapshots can have large velocities:
                        # warmup moves them cm. Restore recorded positions AND
                        # velocities again at the boundary, retaining the real
                        # contact measurements rebuilt by that physics tick.
                        # There are no writes after this boundary.
                        ids=torch.tensor([i],device=env.device);rob=env.scene['robot'];cab=env.scene['cabinet'];j=door_index(env)
                        rq=torch.as_tensor(s['robot'][:9],device=env.device)[None];rv=torch.as_tensor(s['robot'][9:18],device=env.device)[None]
                        rob.write_joint_state_to_sim(rq,rv,env_ids=ids)
                        cq=cab.data.joint_pos[ids].clone();cv=cab.data.joint_vel[ids].clone();cq[:,j]=float(s['gt'][0]);cv[:,j]=float(s['gt'][1])
                        cab.write_joint_state_to_sim(cq,cv,env_ids=ids)
                        env.sim.forward();env.scene['ee_frame'].reset(ids);env.scene['ee_frame'].update(0.,force_recompute=True)
                        robot=robot_observation(env).clone();gt=env.get_tool_state().clone()
                        nr[i]=robot[i];ng[i]=gt[i];active[i]=True;start[i]=gt[i,0]
                        dist=float((robot[i,18:21]-gt[i,2:5]).norm())
                        poserr=float((robot[i,18:21]-torch.as_tensor(s['robot'][18:21],device=env.device)).norm())
                        qerr=float((robot[i,:9]-torch.as_tensor(s['robot'][:9],device=env.device)).abs().max())
                        herr=float((gt[i,2:5]-torch.as_tensor(s['gt'][2:5],device=env.device)).norm())
                        initial[i]=dict(initial_distance=dist,initial_contact=int((gt[i,9:11]>.5).all()),
                                        initially_near=int(dist<.06),handoff_tick=int(env.episode_length_buf[i]),
                                        prefix_eef_error_m=poserr,prefix_q_error_rad=qerr,prefix_handle_error_m=herr)
                        prefix_errors.append(poserr)
                        initialized[s['id']]=(robot[i].cpu().numpy().copy(),gt[i].cpu().numpy().copy())
                        if poserr>.003 or herr>.003 or qerr>.001:raise RuntimeError(f'Physical snapshot initialization mismatch {i}: {initial[i]}')
                        pending.remove(i)
                distance=(nr[:,18:21]-ng[:,2:5]).norm(dim=-1);contact=(ng[:,9:11]>.5).all(-1)
                width=nr[:,7:9].sum(-1)
                conditions=torch.stack((distance<.06,contact,contact&(distance<.06)&(action[:,6]<0)&(width>.005)&(width<.079),ng[:,0]>start+.05),-1)
                runs=torch.where(conditions&policy_active[:,None],runs+1,torch.zeros_like(runs));flags|=(runs>=torch.tensor([6,3,3,30],device=env.device))&policy_active[:,None]
                maxangle=torch.where(active,torch.maximum(maxangle,ng[:,0]),maxangle);length+=policy_active
                for i in done.nonzero().flatten().tolist():
                    if i in selected and bool(active[i]) and i not in complete:
                        s=selected[i];row=dict(cohort_id=s['id'],clone=i,source_clone=s['clone'],assignment=s['assignment'],variant=meta['variant'],seed=meta['seed'],
                             success=int(ng[i,0]>1),reapproach=int(flags[i,0]),recontact=int(flags[i,1]),regrasp=int(flags[i,2]),
                             progress_resumed=int(flags[i,3]),max_angle_rad=float(maxangle[i]),final_angle_rad=float(ng[i,0]),
                             recovery_ticks=int(length[i]),**initial[i])
                        rows.append(row);complete.add(i);active[i]=False
                interactions+=32
                if len(complete)==len(selected):break
            if len(complete)!=len(selected):raise RuntimeError('Missing pressure endpoints')
    finally:env._sample=original_sample
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    names=sorted(initialized)
    np.savez_compressed(path.with_suffix('.initial.npz'),ids=np.asarray(names),robot=np.stack([initialized[k][0] for k in names]),state=np.stack([initialized[k][1] for k in names]))
    with path.with_suffix('.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(variant=meta['variant'],model_seed=meta['seed'],episodes=len(rows),interactions=interactions,
                 cohort=str(cohort),cohort_exact_clone_slots=False,cohort_packed32_layout=True,
                 test_snapshot_reconstruction=True,neutral_contact_warmup_ticks=1,restore_snapshot_after_warmup=True,physical_handoff_tolerance_m=.003,expert_used=False,
                 prefix_eef_error_max_m=max(prefix_errors),mode=meta['mode'])
    for metric in ['success','reapproach','recontact','regrasp','progress_resumed','max_angle_rad','final_angle_rad']:
        summary[metric]=float(np.mean([r[metric] for r in rows]))
    summary['reapproach_when_needed']=float(np.mean([r['reapproach'] for r in rows if not r['initially_near']])) if any(not r['initially_near'] for r in rows) else None
    summary['recontact_when_needed']=float(np.mean([r['recontact'] for r in rows if not r['initial_contact']])) if any(not r['initial_contact'] for r in rows) else None
    path.write_text(json.dumps(summary,indent=2));return summary
