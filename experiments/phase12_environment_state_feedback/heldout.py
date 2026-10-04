"""All state-feedback diagnostics are unconditional, on independent reset seeds."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--strength',required=True)
p.add_argument('--seed',type=int,required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(headless=True).app
import json,os
import h5py,numpy as np,torch,isaaclab_tasks
from randomized_env.door_randomization import create
from experiments.phase12_environment_state_feedback.agent import MeasuredSAC
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,CKPT,prepared,STRENGTHS,run_name
from experiments.phase12_environment_state_feedback.metrics import evaluate
from experiments.phase12_environment_state_feedback.conditions import CONDITIONS
from experiments.phase12_environment_state_feedback.sensitivity import sensitivity
def atomic(path,result):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));os.replace(temp,path)
def main():
 protocol=prepared();folder=OUT/'heldout';folder.mkdir(exist_ok=True)
 name=run_name(args.arm,args.strength,args.seed);path=folder/f'{name}.json'
 if path.exists():raise RuntimeError('Independent tests already exist')
 source=torch.load(ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{args.seed}'/'step_300000.pt',map_location='cuda:0',weights_only=False)
 anchor=torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt',map_location='cuda:0',weights_only=False)
 agent=MeasuredSAC(source,anchor,args.arm,'cuda:0',STRENGTHS[args.strength]);env=create(32,'cuda:0',210000+args.seed,2)
 with h5py.File(ROOT/'datasets'/OUT.name/f'{name}.h5','r') as h:
  # Predetermined sample locations, equal transition counts; no success selection.
  indices=np.linspace(0,299999,1024).astype(int)
  observations=torch.as_tensor(h['transitions/robot'][indices],device=env.device)
 results=[];probes={}
 masks=[] if args.arm=='A' else ['door_angle','door_angular_velocity','progress','remaining_angle','angle_family','all_feedback']
 if args.arm in ('C','D'):masks+=['contact_state']
 if args.arm=='D':masks+=['handle_position']
 try:
  for endpoint,checkpoint in [('best',CKPT/name/'best.pt'),('final',CKPT/name/'step_300000.pt')]:
   saved=torch.load(checkpoint,map_location=env.device,weights_only=False);agent.load_state(saved);agent.actor.eval();agent.actor.cap=.01
   probes[endpoint]=sensitivity(agent.actor,observations,args.arm)
   tests=[(c,'policy',None) for c in CONDITIONS]+[(c,'deterministic',None) for c in ('nominal','level2')]+[('level2','policy',mask) for mask in masks]
   for condition,mode,mask in tests:
    rounds=4 if condition=='level2' and mode=='policy' else 2
    row=evaluate(agent.actor,agent.anchor,env,args.arm,protocol['handle_center'],210000+args.seed,condition,mode,rounds,mask)
    row.update(endpoint=endpoint,arm=args.arm,strength=args.strength,seed=args.seed,ablation=mask,checkpoint_step=saved['env_steps'])
    results.append(row);atomic(folder/f'{name}.partial.json',dict(completed=False,results=results))
    print(name,endpoint,condition,mode,mask,row['metrics']['success'],flush=True)
  atomic(path,dict(completed=True,arm=args.arm,strength=args.strength,seed=args.seed,results=results,sensitivity=probes,
    total_episodes=sum(r['metrics']['episodes'] for r in results),total_interactions=sum(r['metrics']['eval_env_steps'] for r in results)))
 finally:env.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  (OUT/f'heldout_error_{args.arm}_{args.strength}_seed{args.seed}.txt').write_text(traceback.format_exc());raise
 finally:app.close()
