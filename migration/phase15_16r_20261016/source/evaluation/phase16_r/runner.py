"""Atomic Normal-only physical evaluation/training, in independent namespaces."""
import argparse,os,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--job',required=True);parser.add_argument('--socket',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(headless=True).app
import csv,json,time,hashlib
import h5py,numpy as np,torch,isaaclab_tasks
from door_env.door import robot_observation,privileged_state
from door_env.isaac_env import transition_after_step
from experiments.phase14_2_recovery_bc.model import measured
from experiments.phase16_r_residual_sac.common import ROOT,EXP,RES,protocol,checkpoint,save,sha,utc
from evaluation.phase16_r.env import create
from evaluation.phase16_r.chunk import ChunkClient
from algorithms.residual_sac.sac import SAC,Replay,compose
from algorithms.residual_sac.gates import numerical_gate

P=protocol();N=P['resources']['physical_envs']

class TrialStopped(RuntimeError):pass

def raw_state(env):
    return torch.cat((robot_observation(env),measured(env)),-1).detach().cpu().numpy()

def age_state(env):return env.episode_length_buf.detach().cpu().numpy().astype(int)

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Trace:
    def __init__(self,path):
        self.h=h5py.File(path,'x');self.pending={};self.size=0
        self.h.attrs['scope']='real physical ticks; base/sampled residual/actual combined commands saved separately'
    def append(self,**row):
        for k,v in row.items():self.pending.setdefault(k,[]).append(np.asarray(v))
        if len(next(iter(self.pending.values())))>=64:self.flush()
    def flush(self):
        for k,items in self.pending.items():
            a=np.stack(items);n=len(a)
            if k not in self.h:self.h.create_dataset(k,shape=(0,*a.shape[1:]),maxshape=(None,*a.shape[1:]),dtype=a.dtype,chunks=True,compression='gzip',compression_opts=1)
            d=self.h[k];d.resize(self.size+n,axis=0);d[self.size:]=a
        if self.pending:self.size+=len(next(iter(self.pending.values())))
        self.pending={};self.h.flush()
    def close(self):self.flush();self.h.close()

def noise_for(seed,episode,age):
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed,i,int(e),int(t)])).normal(size=6).astype(np.float32) for i,(e,t) in enumerate(zip(episode,age))])

def actions(client,ctx,base,arm,seed,age,sac=None,det=False):
    if arm=='A':
        delta=np.zeros((N,6),np.float32);executed=base.copy();unit=delta.copy();lp=np.zeros((N,1),np.float32);clip=np.zeros((N,6),bool)
    elif arm=='zero':
        with torch.no_grad():
            values=sac.policy(torch.as_tensor(ctx,device=sac.device),torch.as_tensor(base,device=sac.device),deterministic=True)
        delta=values['delta'].cpu().numpy();executed=values['executed'].cpu().numpy();unit=values['unit'].cpu().numpy();lp=values['logp'].cpu().numpy();clip=(values['proposed'].abs()>1).cpu().numpy()
        if np.any(delta):raise RuntimeError('Strict zero actor has nonzero residual')
    elif arm in ['B','small','medium']:
        vector=P['residual']['fixed_B'] if arm=='B' else P['residual']['small_bound'] if arm=='small' else P['residual']['medium_gate_bound']
        delta=np.tile(np.asarray(vector,np.float32),(N,1));unit=delta.copy()
        proposed=base[:,:6]+delta;executed=np.concatenate((np.clip(proposed,-1,1),base[:,6:7]),-1);clip=np.abs(proposed)>1;lp=np.zeros((N,1),np.float32)
    else:
        with torch.no_grad():
            values=sac.policy(torch.as_tensor(ctx,device=sac.device),torch.as_tensor(base,device=sac.device),deterministic=det,noise=torch.as_tensor(noise_for(seed,client.episode,age),device=sac.device))
        delta=values['delta'].cpu().numpy();executed=values['executed'].cpu().numpy();unit=values['unit'].cpu().numpy();lp=values['logp'].cpu().numpy();clip=(values['proposed'].abs()>1).cpu().numpy()
    if not np.isfinite(executed).all() or np.any(np.abs(executed)>1) or not np.array_equal(executed[:,6],base[:,6]):raise RuntimeError('Action/gripper interface stop')
    return delta,executed,unit,lp,clip

def reset(env,client,level,seed):
    env.set_randomization(level);torch.manual_seed(seed);env.reset(seed=seed);client.configure()
    client.seed=P['seeds']['pi_noise_base']+seed*7
    raw=raw_state(env);age=age_state(env)
    if np.any(age):raise RuntimeError('Reset clock not zero')
    return raw,age

def fresh_scene(env):
    # End the previous independent rollout's whole physics scene. Reusing only
    # reset(q,qd) left a measured first-transition history effect in attempts1/2.
    # Inference stays persistent; no physics tick or scored episode is inserted.
    import omni.usd
    # The native standalone STOP callback otherwise loops rendering until PLAY.
    # Disable that app-control callback solely while discarding this scene.
    env.sim._disable_app_control_on_stop_handle=True
    env.sim.stop();env.close();omni.usd.get_context().new_stage()
    return create(N,args.device or 'cuda:0',P['seeds']['engineering_reset'])

def rollout(env,client,out,arm,modelseed,resetseed,nscored,level=2,sac=None,det=False,max_ticks=None):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    raw,age=reset(env,client,level,resetseed)
    if resetseed==P['seeds']['engineering_reset']:
        rob=env.scene['robot'];cab=env.scene['cabinet'];term=env.action_manager.get_term('arm_action')
        save(out.with_suffix('.reset_control.json'),dict(robot_q_target=rob.data.joint_pos_target.cpu().numpy().tolist(),robot_qd_target=rob.data.joint_vel_target.cpu().numpy().tolist(),cabinet_q_target=cab.data.joint_pos_target.cpu().numpy().tolist(),native_jacobians=rob.root_physx_view.get_jacobians().cpu().numpy().tolist(),ik_ee_pos_des=term._ik_controller.ee_pos_des.cpu().numpy().tolist(),ik_ee_quat_des=term._ik_controller.ee_quat_des.cpu().numpy().tolist()))
    ctx,base,cursor=client.peek(raw,age)
    quota=np.full(N,nscored//N);quota[:nscored%N]+=1
    count=np.zeros(N,int);rows=[];starts=[];peak=raw[:,26].copy()
    actual30=np.full(N,np.nan);initial=raw.copy();parameters=env.parameters.detach().cpu().numpy().copy()
    trace=Trace(out.with_suffix('.h5'));ticks=0;totalresets=0;begin=time.perf_counter();firstcalls=len(client.calls)
    try:
        while np.any(count<quota) and (max_ticks is None or ticks<max_ticks):
            delta,action,unit,logp,clip=actions(client,ctx,base,arm,P['seeds']['residual_noise_base']+resetseed+modelseed*10000,age,sac,det)
            client.commit(age)
            _,reward,term,trunc,_=env.step(torch.as_tensor(action,device=env.device))
            rn,gn,done=transition_after_step(env,term,trunc)
            ngt=gn.cpu().numpy();nr=rn.cpu().numpy();d=done.cpu().numpy()
            reward=reward.cpu().numpy()
            signed=reward-20*np.maximum(ngt[:,1],0)*env.step_dt+20*(ngt[:,0]-raw[:,26])
            peak=np.maximum(peak,ngt[:,0])
            at30=age==29;actual30[at30]=ngt[at30,0]
            trace.append(raw=raw,next_robot=nr,next_gt=ngt,base=base,delta=delta,unit=unit,executed=action,logp=logp,clip=clip,age=age,episode=client.episode.copy(),cursor=cursor,chunk=client.cache.copy(),reward_original=reward,reward_signed=signed,done=d,term=term.cpu().numpy(),trunc=trunc.cpu().numpy(),scored=count<quota)
            for i in np.flatnonzero(d):
                if count[i]<quota[i]:
                    row=dict(clone=int(i),episode=int(count[i]),success=int(ngt[i,0]>1),ticks=int(round(nr[i,25]*600)),final_angle=float(ngt[i,0]),peak_angle=float(peak[i]),requested_angle=float(parameters[i,0]),actual_angle_before_policy=float(initial[i,26]),actual_angle_tick30=float(actual30[i]) if np.isfinite(actual30[i]) else None,offset_x=float(parameters[i,1]),offset_y=float(parameters[i,2]),offset_z=float(parameters[i,3]),terminal='success' if ngt[i,0]>1 else 'timeout' if bool(trunc[i]) else 'unsafe')
                    rows.append(row);starts.append(dict(clone=int(i),episode=int(count[i]),raw=initial[i].tolist(),parameters=parameters[i].tolist()))
                    count[i]+=1
            totalresets+=int(d.sum());ticks+=1
            live=raw_state(env);newage=age_state(env)
            ids=np.flatnonzero(d)
            if len(ids):
                client.reset_ids(ids);initial[ids]=live[ids];parameters[ids]=env.parameters.cpu().numpy()[ids];peak[ids]=live[ids,26];actual30[ids]=np.nan
                if np.any(newage[ids]!=0):raise RuntimeError('Auto-reset failed')
            raw,age=live,newage
            ctx,base,cursor=client.peek(raw,age)
    finally:trace.close()
    calls=client.calls[firstcalls:]
    result=dict(utc=utc(),arm=arm,model_seed=modelseed,reset_seed=resetseed,episodes=len(rows),scored_target=nscored,success=float(np.mean([r['success'] for r in rows])) if rows else None,physical_interactions=ticks*N,vector_ticks=ticks,automatic_resets=totalresets,elapsed_s=time.perf_counter()-begin,distribution='sampled' if not det else 'mean_residual',level=level,checkpoint_sha256=sha(checkpoint(modelseed)),calls=calls,trace_sha256=sha(out.with_suffix('.h5')),nonscored_physical_interactions_included_in_cost=True)
    write_csv(out.with_suffix('.csv'),rows);save(out.with_suffix('.initial.json'),starts);save(out.with_suffix('.json'),result)
    print('EVAL_COMPLETE '+json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
    return result

def engineering(env,client,out,seed):
    out.mkdir(parents=True,exist_ok=True)
    values={}
    for arm in ['A','zero']:
        if arm!='A':env=fresh_scene(env)
        zero=SAC(seed,P['residual']['small_bound'],env.device,P['sac']) if arm=='zero' else None
        values[arm]=rollout(env,client,out/arm,arm,seed,P['seeds']['engineering_reset'],N,sac=zero)
    errors={}
    with h5py.File(out/'A.h5') as x,h5py.File(out/'zero.h5') as y:
        for key in ['base','executed','raw','next_robot','next_gt','reward_original','reward_signed','done','cursor','chunk']:
            if x[key].shape!=y[key].shape:raise RuntimeError('Zero physical trace length mismatch')
            a=x[key][:];b=y[key][:];diff=float(np.max(np.abs(a.astype(float)-b.astype(float))))
            errors[key]=diff
            if diff>1e-6:raise RuntimeError('Zero equivalence failed '+key+' '+str(diff))
        assert values['A']['success']==values['zero']['success']
    for arm in ['small','medium']:
        env=fresh_scene(env)
        rollout(env,client,out/arm,arm,seed,P['seeds']['engineering_reset'],N,max_ticks=60)
    with h5py.File(out/'A.h5') as ref:
        for arm in ['small','medium']:
            with h5py.File(out/(arm+'.h5')) as cand:
                if np.array_equal(cand['executed'][0],ref['executed'][0]):raise RuntimeError('Residual did not change command')
                if np.max(np.abs(cand['next_robot'][0,:,:7]-ref['next_robot'][0,:,:7]))<1e-8:raise RuntimeError('Residual did not change physics')
                assert np.array_equal(cand['executed'][:,:,6],cand['base'][:,:,6])
                errors[arm+'_joint_delta_tick1_max']=float(np.max(np.abs(cand['next_robot'][0,:,:7]-ref['next_robot'][0,:,:7])))
        raw=ref['raw'][:-1].reshape(-1,39);base=ref['base'][:-1].reshape(-1,7)
        chunk=ref['chunk'][:-1].reshape(-1,70);cur=ref['cursor'][:-1].reshape(-1)
        next_raw=ref['raw'][1:].reshape(-1,39);next_base=ref['base'][1:].reshape(-1,7)
        next_chunk=ref['chunk'][1:].reshape(-1,70);next_cur=ref['cursor'][1:].reshape(-1)
        context=np.concatenate((2/np.pi*np.arctan((raw-client.mean)/client.std),base,chunk,np.eye(5)[cur]),-1).astype(np.float32)
        next_context=np.concatenate((2/np.pi*np.arctan((next_raw-client.mean)/client.std),next_base,next_chunk,np.eye(5)[next_cur]),-1).astype(np.float32)
        done=ref['done'][:-1].reshape(-1,1);next_context[done[:,0]]=0;next_base[done[:,0]]=0
        ids=np.r_[np.random.default_rng(160123).integers(0,len(raw),240),np.flatnonzero(done[:,0])[:16]]
        batch=dict(context=context[ids],base=base[ids],next_context=next_context[ids],next_base=next_base[ids],delta=ref['delta'][:-1].reshape(-1,6)[ids],executed=ref['executed'][:-1].reshape(-1,7)[ids],reward=ref['reward_original'][:-1].reshape(-1,1)[ids],done=done[ids].astype(np.float32))
        gate=numerical_gate({k:torch.as_tensor(v,device=env.device) for k,v in batch.items()},env.device)
        save(out/'sac_real_gate.json',dict(utc=utc(),**gate,real_source='engineering frozen-A physical transitions, next base/context from following actual tick',source_trace_sha256=sha(out/'A.h5'),physical_interactions=values['A']['physical_interactions'],main_trial_updates=0))
    save(out/'physical_gate.json',dict(utc=utc(),passed=True,errors=errors,zero_success=values['A']['success'],small_medium_not_performance_selection=True))
    print('PHYSICAL_GATE_PASSED',flush=True)
    return env

def reset_probe(env,client,out,seed):
    out.mkdir(parents=True,exist_ok=True)
    def snapshot():
        rob=env.scene['robot'];cab=env.scene['cabinet'];term=env.action_manager.get_term('arm_action')
        values=dict(raw=raw_state(env),robot_q=rob.data.joint_pos,robot_qd=rob.data.joint_vel,
            native_q=rob.root_physx_view.get_dof_positions(),native_qd=rob.root_physx_view.get_dof_velocities(),
            body_pos=rob.data.body_pos_w,body_quat=rob.data.body_quat_w,root=rob.data.root_state_w,
            native_links=rob.root_physx_view.get_link_transforms(),native_jacobian=rob.root_physx_view.get_jacobians(),
            robot_target=rob.data.joint_pos_target,robot_velocity_target=rob.data.joint_vel_target,
            cabinet_target=cab.data.joint_pos_target,ik_position=term._ik_controller.ee_pos_des,
            ik_quat=term._ik_controller.ee_quat_des,processed=term.processed_actions)
        pos,quat=term._compute_frame_pose();values.update(ik_current_position=pos,ik_current_quat=quat)
        return {k:v.detach().cpu().numpy().copy().tolist() if torch.is_tensor(v) else v.tolist() for k,v in values.items()}
    action=None
    for iteration in range(3):
        raw,age=reset(env,client,2,P['seeds']['engineering_reset']);ctx,base,cur=client.peek(raw,age)
        if action is None:action=base.copy()
        record=dict(iteration=iteration,before=snapshot(),action=action.tolist(),base=base.tolist())
        tensor=torch.as_tensor(action,device=env.device)
        env.action_manager.process_action(tensor);record['processed']=snapshot()
        env.action_manager.apply_action();record['applied']=snapshot()
        env.scene.write_data_to_sim();record['submitted']=snapshot()
        # These calls probe controller state without stepping physics. The actual
        # env.step recomputes the same command using the unchanged physical pose.
        _,rew,term,trunc,_=env.step(tensor)
        record['after']=snapshot();record['reward']=rew.cpu().numpy().tolist()
        save(out/f'reset{iteration}.json',record)
    save(out/'complete.json',dict(utc=utc(),passed=True,diagnostic_only=True,physical_interactions=3*N,gate_passed=False))

def baseline(env,client,out,seed):
    out.mkdir(parents=True,exist_ok=True)
    first=True
    for arm in ['A','B']:
        for scope,num,level,resetseed in [('random',128,2,P['seeds']['test_reset']),('validation',64,2,P['seeds']['validation_reset']),('fixed',32,0,P['seeds']['fixed_reset'])]:
            if not first:env=fresh_scene(env)
            first=False
            rollout(env,client,out/f'{arm}_{scope}',arm,seed,resetseed,num,level)
    save(out/'complete.json',dict(utc=utc(),passed=True,seed=seed,rl_updates=0))
    return env

def train(env,client,out,seed,arm):
    out.mkdir(parents=True,exist_ok=True)
    ckdir=ROOT/f'checkpoints/phase16_r/{arm}_seed{seed}';ckdir.mkdir(parents=True,exist_ok=True)
    sac=SAC(seed,P['residual']['small_bound'],env.device,P['sac'])
    torch.save(sac.state(),ckdir/'initial.pt')
    replay=Replay(50000);rng=np.random.default_rng(seed+16300000)
    raw,age=reset(env,client,2,P['seeds']['train_reset_base']+seed)
    ctx,base,cursor=client.peek(raw,age);trace=Trace(out/'training.h5')
    from torch.utils.tensorboard import SummaryWriter
    tb=SummaryWriter(str(out/'tensorboard'))
    steps=0;rows=[];episodes=[];nodes=iter(P['evaluation']['nodes'][1:]);nextnode=next(nodes,None)
    cost=dict(training_interactions=0,training_automatic_resets=0,successful_online_episodes=0,optimizer_updates=0,gate_interactions=0)
    started=time.perf_counter();update_wall=0;pause_wall=0;physics_wall=0
    def node(step):
        nonlocal pause_wall
        dest=ckdir/f'step{step}.pt';torch.save(sac.state(),dest)
        req=dict(utc=utc(),arm=arm,seed=seed,step=step,checkpoint=str(dest.relative_to(ROOT)),checkpoint_sha256=sha(dest),training_interactions=steps,optimizer_updates=sac.updates)
        save(out/f'node{step}.json',req)
        status=out/f'node{step}.validation_done.json';wait=time.monotonic()
        while not status.exists():
            if (out/'stop.json').exists():
                pause_wall+=time.monotonic()-wait
                raise TrialStopped('Validation stop '+(out/'stop.json').read_text())
            if time.monotonic()-wait>7200:raise RuntimeError('Evaluation coordination timeout')
            time.sleep(2)
        v=json.loads(status.read_text())
        pause_wall+=time.monotonic()-wait
        if not v.get('passed'):raise TrialStopped('Independent node evaluation failed '+str(v))
    try:
        node(0)
        while steps<50000:
            delta,action,unit,logp,clip=actions(client,ctx,base,arm,P['seeds']['residual_noise_base']+seed,age,sac)
            context=ctx.copy();currentbase=base.copy();currentraw=raw.copy();currentage=age.copy();cur=cursor.copy();chunk=client.cache.copy();ep=client.episode.copy()
            client.commit(age)
            physics_begin=time.perf_counter()
            _,reward,term,trunc,_=env.step(torch.as_tensor(action,device=env.device))
            physics_wall+=time.perf_counter()-physics_begin
            rn,gn,done=transition_after_step(env,term,trunc);nr=rn.cpu().numpy();ng=gn.cpu().numpy();d=done.cpu().numpy()
            original=reward.cpu().numpy();signed=original-20*np.maximum(ng[:,1],0)*env.step_dt+20*(ng[:,0]-raw[:,26]);selected=original if arm=='C' else signed
            live=raw_state(env);newage=age_state(env);ids=np.flatnonzero(d)
            for i in ids:
                row=dict(clone=int(i),episode=int(ep[i]),training_interactions=steps+N,success=int(ng[i,0]>1),ticks=int(round(nr[i,25]*600)),final_angle=float(ng[i,0]),terminal='success' if ng[i,0]>1 else 'timeout' if bool(trunc[i]) else 'unsafe')
                episodes.append(row);cost['successful_online_episodes']+=row['success']
            if len(ids):
                client.reset_ids(ids)
                if np.any(newage[ids]!=0):raise RuntimeError('Auto-reset clock failed')
            raw,age=live,newage;ctx,base,cursor=client.peek(raw,age)
            nextctx=ctx.copy();nextbase=base.copy()
            # Terminal physical state is saved; no terminal base inference exists.
            # Terminal next context/base are explicitly zero and never bootstrap.
            nextctx[d]=0;nextbase[d]=0
            replay.add(context=context,base=currentbase,delta=delta,executed=action,unit=unit,logp=logp,next_context=nextctx,next_base=nextbase,reward=selected[:,None],done=d[:,None],raw=currentraw,next_robot=nr,next_gt=ng,age=currentage[:,None],next_age=np.rint(nr[:,25]*600)[:,None],live_reset_age=newage[:,None],cursor=cur[:,None],next_cursor=cursor[:,None],episode=ep[:,None],original_reward=original[:,None],signed_reward=signed[:,None])
            trace.append(raw=currentraw,next_robot=nr,next_gt=ng,base=currentbase,delta=delta,unit=unit,executed=action,logp=logp,clip=clip,age=currentage,episode=ep,cursor=cur,chunk=chunk,reward_original=original,reward_signed=signed,done=d,term=term.cpu().numpy(),trunc=trunc.cpu().numpy())
            steps+=N;cost['training_interactions']=steps;cost['training_automatic_resets']+=len(ids)
            if steps>=2048:
                for _ in range(N//8):
                    begin=time.perf_counter();m=sac.update(replay.sample(256,rng,env.device));update_wall+=time.perf_counter()-begin
                rows.append(dict(interactions=steps,updates=sac.updates,**m))
                for k,v in m.items():tb.add_scalar(k,v,steps)
            if steps%512==0 or steps==50000:
                save(out/'progress.json',dict(utc=utc(),phase='training',arm=arm,seed=seed,steps=steps,updates=sac.updates,online_successes=cost['successful_online_episodes'],resets=cost['training_automatic_resets'],elapsed_s=time.perf_counter()-started,update_wall_s=update_wall))
                print('TRAIN_PROGRESS '+json.dumps(dict(arm=arm,seed=seed,steps=steps,updates=sac.updates,successes=cost['successful_online_episodes'])),flush=True)
            if nextnode is not None and steps==nextnode:
                trace.flush();replay.save(out/f'replay_step{steps}.npz');node(steps);nextnode=next(nodes,None)
        replay.save(out/'replay_final.npz')
        cost.update(optimizer_updates=sac.updates,update_wall_s=update_wall,physics_step_wall_s=physics_wall,evaluation_pause_wall_s=pause_wall,active_wall_s=time.perf_counter()-started-pause_wall,elapsed_s=time.perf_counter()-started,train_rng_seed=seed+16300000,ongoing_episodes_at_budget_end='right-censored, retained in raw replay, never counted as failures or successes')
        save(out/'complete.json',dict(utc=utc(),passed=True,arm=arm,seed=seed,**cost))
    except TrialStopped as error:
        replay.save(out/'replay_stopped.npz');torch.save(sac.state(),ckdir/f'stopped_step{steps}.pt')
        cost.update(optimizer_updates=sac.updates,elapsed_s=time.perf_counter()-started,evaluation_pause_wall_s=pause_wall,active_wall_s=time.perf_counter()-started-pause_wall,update_wall_s=update_wall,physics_step_wall_s=physics_wall)
        save(out/'stopped.json',dict(utc=utc(),passed=False,controlled_trial_stop=True,arm=arm,seed=seed,reason=str(error),**cost))
    finally:
        write_csv(out/'updates.csv',rows);write_csv(out/'online_episodes.csv',episodes)
        save(out/'inference_calls.json',dict(calls=client.calls,scope='actual training base queries, including startup and graph gates in observed service scope'))
        trace.close();tb.close()

def evalnode(env,client,out,job):
    out.mkdir(parents=True,exist_ok=True)
    seed=job['seed'];arm=job['arm'];step=job['step']
    state=torch.load(ROOT/job['checkpoint'],map_location=env.device,weights_only=False)
    assert sha(ROOT/job['checkpoint'])==job['checkpoint_sha256']
    sac=SAC(seed,P['residual']['small_bound'],env.device,P['sac']);sac.restore(state)
    rollout(env,client,out/'validation_sampled',arm,seed,P['seeds']['validation_reset'],64,sac=sac)
    env=fresh_scene(env)
    rollout(env,client,out/'validation_mean',arm,seed,P['seeds']['validation_reset'],64,sac=sac,det=True)
    if step==50000 or job.get('final_test'):
        for scope,num,level,resetseed in [('random',128,2,P['seeds']['test_reset']),('fixed',32,0,P['seeds']['fixed_reset'])]:
            for det in [False,True]:
                env=fresh_scene(env)
                rollout(env,client,out/f'{scope}_{"mean" if det else "sampled"}',arm,seed,resetseed,num,level,sac,det)
    save(out/'complete.json',dict(utc=utc(),passed=True,seed=seed,arm=arm,step=step,checkpoint_sha256=sha(ROOT/job['checkpoint'])))
    return env

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    job=json.loads(Path(args.job).read_text());out=ROOT/job['output'];out.mkdir(parents=True,exist_ok=True)
    env=create(N,args.device or 'cuda:0',P['seeds']['engineering_reset'])
    client=ChunkClient(args.socket,N,P['seeds']['pi_noise_base']);client.load(checkpoint(job['seed']))
    try:
        if job['mode']=='engineering':env=engineering(env,client,out,job['seed'])
        elif job['mode']=='reset_probe':reset_probe(env,client,out,job['seed'])
        elif job['mode']=='baseline':env=baseline(env,client,out,job['seed'])
        elif job['mode']=='train':train(env,client,out,job['seed'],job['arm'])
        elif job['mode']=='evalnode':env=evalnode(env,client,out,job)
        elif job['mode']=='eval_rollout':
            # A/B allocate the identical untrained residual model for the same
            # CUDA allocation footprint, while executing the frozen control law.
            sac=SAC(job['seed'],P['residual']['small_bound'],env.device,P['sac'])
            if job.get('checkpoint'):
                assert sha(ROOT/job['checkpoint'])==job['checkpoint_sha256']
                sac.restore(torch.load(ROOT/job['checkpoint'],map_location=env.device,weights_only=False))
            rollout(env,client,out/job['casefile'],job['arm'],job['seed'],job['resetseed'],job['nscored'],job['level'],sac,job.get('det',False),job.get('max_ticks'))
        else:raise ValueError(job['mode'])
        complete_name=job['casefile']+'.worker_complete.json' if job['mode']=='eval_rollout' else 'worker_complete.json'
        save(out/complete_name,dict(utc=utc(),passed=True,pid=os.getpid(),inference_status=client.rpc(dict(op='status')),checkpoint_sha256=sha(checkpoint(job['seed']))))
    finally:
        if job.get('keep_inference'):client.disconnect()
        else:client.stop()
        env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        job=json.loads(Path(args.job).read_text());name=job.get('casefile','')+'.failed.json' if job['mode']=='eval_rollout' else 'failed.json';save(ROOT/job['output']/name,dict(utc=utc(),error=error,pid=os.getpid()))
        os._exit(1)
    print('WORKER_COMPLETE',flush=True);os._exit(0)
