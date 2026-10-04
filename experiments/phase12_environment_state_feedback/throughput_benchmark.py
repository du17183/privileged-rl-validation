"""Bounded evaluation only. Additional physical interactions are reported.

Use identical actor/anchor/task/reward; count the entire unchanged evaluation
loop so observation, contact monitoring and terminal logging overhead survives.
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--count',type=int,required=True);p.add_argument('--label',default=None);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(headless=True).app
import json,time,torch,isaaclab_tasks
from randomized_env.door_randomization import create
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,CKPT,prepared
from experiments.phase12_environment_state_feedback.agent import MeasuredSAC
from experiments.phase12_environment_state_feedback.metrics import evaluate
try:
 source=torch.load(ROOT/'checkpoints/phase9_safe_online/P9C_seed0/step_300000.pt',map_location='cuda:0',weights_only=False)
 old=torch.load(ROOT/'checkpoints/phase8_progress_rl/P8B_seed0/best.pt',map_location='cuda:0',weights_only=False)
 agent=MeasuredSAC(source,old,'D','cuda:0');agent.load_state(torch.load(CKPT/'P12D_strong_seed0/step_300000.pt',map_location='cuda:0',weights_only=False))
 env=create(args.count,'cuda:0',821000,2);torch.cuda.synchronize();start=time.perf_counter()
 result=evaluate(agent.actor,agent.anchor,env,'D',prepared()['handle_center'],821000,'level2','policy',1)
 torch.cuda.synchronize();duration=time.perf_counter()-start
 out=dict(count=args.count,seconds=duration,metrics=result['metrics'],env_steps_per_second=result['metrics']['eval_env_steps']/duration,
   episodes_per_second=args.count/duration,allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
   training_interactions=0,scope='Throughput benchmark, excluded from comparisons and replay')
 path=OUT/'throughput';path.mkdir(exist_ok=True);(path/f'{args.label or "batch_"+str(args.count)}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);env.close()
finally:app.close()
