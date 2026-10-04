"""Parallel independent 32-clone tests, preserving original physical plans.

Each group retains the original 32 clone RNG streams and sequential episode
quota. One 32x7 CUDA normal draw is repeated across groups at each tick, exactly
the common seed/draw shape of serial tests. No episode count is increased.
"""
import math,time,os,cProfile,pstats
import numpy as np
import torch
from randomized_env.door_randomization import RandomizedDoorEnv,LEVELS
from progress_rl.door_env import config
from door_env.door import robot_observation
from progress_rl.progress_monitor import EpisodeMonitor,aggregate
from safe_online.kl_constraint import normal_kl
from environment_feedback.normalization import encode
from experiments.phase12_environment_state_feedback.data import transition
from experiments.phase12_environment_state_feedback.conditions import CONDITIONS
from evaluation.stratified_generalization import slice_episodes

class BatchedTestEnv(RandomizedDoorEnv):
 def __init__(self,count,device,seed):
  self.groups=[dict(condition='level2',rounds=0,mode='policy',ablation=None)]*count
  super().__init__(config('B',count*32,device,seed))
  # Match the original create(): initialize progress/reset buffers before
  # enabling pre-reset terminal capture. ManagerBasedRLEnv does not reset here.
  self.reset(seed=seed)
  self.capture_terminal=True
 def seed_streams(self,seed):
  self.random_seed=int(seed)
  self.streams=[np.random.default_rng(np.random.SeedSequence([int(seed),i%32,11011])) for i in range(self.num_envs)]
 def _sample(self,ids):
  values=[]
  for i in ids:
   level,preset=CONDITIONS[self.groups[i//32]['condition']];angle_max,offset,friction=LEVELS[level];rng=self.streams[i]
   row=[math.radians(rng.uniform(0,angle_max)),*rng.uniform(-offset,offset,3),rng.uniform(1-friction,1+friction)]
   if preset:
    if 'angle_deg' in preset:row[0]=math.radians(preset['angle_deg'])
    if 'angle_range_deg' in preset:row[0]=math.radians(rng.uniform(*preset['angle_range_deg']))
    if 'offset_xyz' in preset:row[1:4]=preset['offset_xyz']
    if 'offset_bound' in preset:row[1:4]=list(rng.uniform(-preset['offset_bound'],preset['offset_bound'],3))
    if 'offset_linf_range' in preset:
     lower,upper=preset['offset_linf_range']
     while True:
      draw=rng.uniform(-upper,upper,3)
      if lower<=np.max(np.abs(draw))<=upper:row[1:4]=list(draw);break
    if 'friction_scale' in preset:row[4]=preset['friction_scale']
   values.append(row)
  return torch.as_tensor(values,dtype=torch.float32,device=self.device)

def apply_masks(obs,groups,arm):
 if arm=='A':return obs
 for g,spec in enumerate(groups):
  mask=spec.get('ablation')
  if mask is None:continue
  x=obs[g*32:(g+1)*32]
  if mask in ('door_angle','door_angular_velocity','progress','remaining_angle'):
   x[:,dict(door_angle=26,door_angular_velocity=27,progress=29,remaining_angle=30)[mask]]=0
  elif mask=='contact_state':x[:,31:33]=0
  elif mask=='handle_position':x[:,33:36]=0
  elif mask in ('angle_family','all_feedback'):
   x[:,26]=0;x[:,27]=0;x[:,29]=0;x[:,30]=x[:,28]
   if mask=='all_feedback':
    if arm in ('C','D'):x[:,31:33]=0
    if arm=='D':x[:,33:36]=0
  else:raise ValueError(mask)
 return obs

@torch.no_grad()
def evaluate_batch(actor,anchor,env,arm,center,seed,tests):
 slots=env.num_envs//32
 if len(tests)>slots:raise ValueError('Too many tests')
 padding=[dict(condition='nominal',mode='deterministic',rounds=0,ablation=None)]*(slots-len(tests))
 groups=[dict(s) for s in tests]+padding;env.groups=groups
 torch.manual_seed(seed);env.reset(seed=seed)
 quotas=torch.tensor([s['rounds'] for s in groups],device=env.device).repeat_interleave(32)
 counts=torch.zeros(env.num_envs,dtype=torch.long,device=env.device);lengths=counts.clone()
 returns=torch.zeros(env.num_envs,device=env.device);monitor=EpisodeMonitor(env.num_envs,env.device)
 initial=env.parameters.clone();handles=env.get_environment_state()['handle_position'].clone()
 records=[[] for _ in groups];totals=torch.zeros((slots,5),device=env.device);samples=torch.zeros(slots,device=env.device)
 policy=torch.tensor([s['mode']=='policy' for s in groups],device=env.device).repeat_interleave(32)[:,None]
 group_end=[0 if s['rounds']==0 else None for s in groups];start=time.perf_counter()
 profiler=cProfile.Profile() if os.environ.get('P12_PROFILE')=='1' else None
 if profiler:profiler.enable()
 for tick in range((env.max_episode_length+2)*(int(quotas.max())+1)):
  obs=apply_masks(encode(robot_observation(env),env.get_environment_state(),arm,center),groups,arm)
  mean,log_std=actor.distribution(obs);old_mean,old_std=anchor.distribution(obs)
  noise=torch.randn((32,7),device=env.device).repeat(slots,1)
  action=torch.where(policy,mean+log_std.exp()*noise,mean).tanh()
  active=(counts<quotas).float()
  values=torch.stack((log_std.exp().mean(-1),normal_kl(mean,log_std,old_mean,old_std),
   (mean.tanh()-old_mean.tanh()).square().mean(-1),(log_std+.5*math.log(2*math.pi*math.e)).sum(-1),mean.tanh().abs().mean(-1)),-1)
  totals+=(values*active[:,None]).reshape(slots,32,5).sum(1);samples+=active.reshape(slots,32).sum(1)
  before=env.get_tool_state()[:,:1].clone();_,reward,term,trunc,_=env.step(action)
  _,gt,done=transition(env,term,trunc,arm,center);lengths+=1;returns+=reward
  monitor.update(before,gt[:,:1],(gt[:,9:11]>.5).all(-1,keepdim=True))
  ids=done.nonzero().squeeze(-1)
  if len(ids):
   # One batch CPU transfer replaces many synchronized scalar reads per episode.
   matrix=torch.stack((monitor.maximum[ids],gt[ids,0],monitor.increase[ids],monitor.regression[ids],
    env.final_start[ids,0],env.final_target[ids,0],monitor.max_stall[ids],monitor.max_no_contact[ids],
    lengths[ids],returns[ids],counts[ids],quotas[ids]),-1).cpu().numpy()
   params=initial[ids].cpu().numpy();first_handles=handles[ids].cpu().numpy()
   last_handles=env.final_environment['handle_position'][ids].cpu().numpy();indices=ids.cpu().tolist()
   for j,i in enumerate(indices):
    maximum,angle,increase,regression,begin,target,stall,no_contact,length,ret,index,quota=matrix[j]
    if index>=quota:continue
    p=params[j];row=dict(max_angle=float(maximum),final_angle=float(angle),cumulative_increase=float(increase),
     regression_amount=float(regression),progress=min(1.,max(0.,float((angle-begin)/(target-begin)))),
     success=float(angle>target),regression_event=float(regression>.05),stalled=float(stall>=120),lost_contact=float(no_contact>=120),
     episode_steps=int(length),return_value=float(ret),initial_angle_deg=math.degrees(float(p[0])),offset_x_m=float(p[1]),
     offset_y_m=float(p[2]),offset_z_m=float(p[3]),offset_linf_cm=100*float(np.max(np.abs(p[1:4]))),friction_scale=float(p[4]),
     initial_handle_position=first_handles[j].tolist(),final_handle_position=last_handles[j].tolist(),env_index=i%32,episode_index=int(index))
    records[i//32].append(row)
   counts[ids]+=1;lengths[ids]=0;returns[ids]=0
   for field in (monitor.maximum,monitor.increase,monitor.regression,monitor.stall,monitor.max_stall,monitor.no_contact,monitor.max_no_contact):field[ids]=0
   initial[ids]=env.parameters[ids];handles[ids]=env.get_environment_state()['handle_position'][ids]
  group_done=(counts>=quotas).reshape(slots,32).all(-1).cpu().tolist()
  for g,finished in enumerate(group_done):
   if finished and group_end[g] is None:group_end[g]=tick+1
  if all(group_done):break
  if profiler and tick==49:
   profiler.disable();pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)
   print('profile ticks50',time.perf_counter()-start,'counts',counts.reshape(slots,32).max(-1).values.tolist(),flush=True)
   profiler=None
  if os.environ.get('P12_PROFILE')=='1' and tick%100==0:
   print('batch tick',tick,'seconds',time.perf_counter()-start,'completed_groups',group_done,flush=True)
 values=(totals/samples.clamp_min(1)[:,None]).cpu().numpy();results=[]
 for g,spec in enumerate(tests):
  rows=records[g]
  if len(rows)!=32*spec['rounds']:raise RuntimeError('Incomplete original episode quota')
  scalar_keys=[k for k,v in rows[0].items() if isinstance(v,(float,int))]
  metrics=aggregate([{k:r[k] for k in scalar_keys} for r in rows])
  metrics.update(episodes=len(rows),eval_env_steps=group_end[g]*32,mode=spec['mode'],condition=spec['condition'],
    physical_reset_checks=env.physical_reset_checks)
  for n,k in enumerate(('effective_std','anchor_kl','anchor_action_mse','pre_tanh_entropy','mean_action_abs')):metrics[k]=float(values[g,n])
  metrics['slices']=slice_episodes(rows)
  results.append(dict(metrics=metrics,records=rows,ablation=spec.get('ablation')))
 return results,dict(physical_interactions=env.num_envs*(tick+1),reported_condition_interactions=sum(r['metrics']['eval_env_steps'] for r in results),
     completed_episodes=sum(len(r['records']) for r in results),seconds=time.perf_counter()-start,groups=len(tests),parallel_envs=env.num_envs)
