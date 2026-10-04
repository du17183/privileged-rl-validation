"""Fixed robot state counterfactuals; normalized action units, not success proof."""
import math,torch
@torch.no_grad()
def sensitivity(actor,obs,arm):
 result=dict(observations=len(obs),robot_state_fixed=True,probes=[])
 if arm=='A':return dict(**result,angle_only_pair_l2_mean=0.,coherent_reset_pair_l2_mean=0.)
 for coherent in (False,True):
  pair=[]
  for deg in (0.,5.):
   x=obs.clone();angle=math.radians(deg);x[:,26]=angle
   if coherent:
    x[:,27]=0.;x[:,28]=1.;x[:,29]=0.;x[:,30]=1.-angle
   pair.append(actor(x,deterministic=True)[0])
  delta=pair[1]-pair[0]
  key='coherent_reset_pair' if coherent else 'angle_only_pair'
  result[key+'_l2_mean']=float(delta.norm(dim=-1).mean())
  result[key+'_l2_sd']=float(delta.norm(dim=-1).std())
  result[key+'_rms']=float(delta.square().mean().sqrt())
  result[key+'_mean_action_delta']=delta.mean(0).tolist()
 baseline=actor(obs,deterministic=True)[0]
 for feature,indices in [('door_angle',[26]),('door_angular_velocity',[27]),('progress',[29]),('remaining_angle',[30])]+([('contact_state',[31,32])] if arm in ('C','D') else [])+([('handle_position',[33,34,35])] if arm=='D' else []):
  x=obs.clone();x[:,indices]=0.;delta=actor(x,deterministic=True)[0]-baseline
  result['probes'].append(dict(feature=feature,action_l2_mean=float(delta.norm(dim=-1).mean()),action_rms=float(delta.square().mean().sqrt())))
 result['added_actor_weight_norm']=float(actor.encoder[0].weight[:,26:].norm())
 result['caution']='Angle-only interventions can be physically inconsistent. Coherent reset changes related scalars but holds robot and handle coordinates fixed. Sensitivity is dependence evidence, not appropriate action or generalization evidence.'
 return result
