import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(headless=True).app
import json,time,torch,isaaclab_tasks
from randomized_env.door_randomization import create
from experiments.phase12_environment_state_feedback.batched_evaluation import BatchedTestEnv,evaluate_batch
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,CKPT,prepared
from experiments.phase12_environment_state_feedback.agent import MeasuredSAC
from experiments.phase12_environment_state_feedback.metrics import evaluate
try:
 actor_source=torch.load(ROOT/'checkpoints/phase9_safe_online/P9C_seed0/step_300000.pt',map_location='cuda:0',weights_only=False)
 anchor_source=torch.load(ROOT/'checkpoints/phase8_progress_rl/P8B_seed0/best.pt',map_location='cuda:0',weights_only=False)
 agent=MeasuredSAC(actor_source,anchor_source,'D','cuda:0');agent.load_state(torch.load(CKPT/'P12D_strong_seed0/step_300000.pt',map_location='cuda:0',weights_only=False))
 # Separate Kit invocations: existing serial env is used only in the serial mode.
 if __import__('os').environ.get('P12_PREFLIGHT_SERIAL')=='1':
  env=create(32,'cuda:0',821001,2);rows=[];start=time.perf_counter()
  for c in ('nominal','level2'):
   rows.append(evaluate(agent.actor,agent.anchor,env,'D',prepared()['handle_center'],821001,c,'policy',2))
  result=dict(rows=rows,seconds=time.perf_counter()-start,physical_interactions=sum(r['metrics']['eval_env_steps'] for r in rows))
  (OUT/'throughput/serial_reference.json').write_text(json.dumps(result));env.close()
 else:
  env=BatchedTestEnv(2,'cuda:0',821001)
  specs=[dict(condition=c,mode='policy',rounds=2,ablation=None) for c in ('nominal','level2')]
  start=time.perf_counter();rows,cost=evaluate_batch(agent.actor,agent.anchor,env,'D',prepared()['handle_center'],821001,specs)
  reference=json.loads((OUT/'throughput/serial_reference.json').read_text());checks=[]
  for old,new in zip(reference['rows'],rows):
   def plan(records):return {(r['env_index'],r['episode_index']):[r[k] for k in ('initial_angle_deg','offset_x_m','offset_y_m','offset_z_m','friction_scale')] for r in records}
   plans_match=plan(old['records'])==plan(new['records'])
   delta=abs(old['metrics']['success']-new['metrics']['success'])
   checks.append(dict(condition=new['metrics']['condition'],parameters_exact_match=plans_match,success_delta=delta,
      serial_success=old['metrics']['success'],batched_success=new['metrics']['success']))
  passed=all(r['parameters_exact_match'] and r['success_delta']<=1/32 for r in checks)
  result=dict(passed=passed,checks=checks,batched_cost=cost,serial_seconds=reference['seconds'],batched_seconds=time.perf_counter()-start,
     note='No learning change. Same episode budgets, physical RNG streams, 32x7 common CUDA action draws. Clone origin/floating-point scheduling can differ.',
     serial_interactions=reference['physical_interactions'],batched_interactions=cost['physical_interactions'])
  (OUT/'throughput/batched_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);env.close()
  if not passed:raise RuntimeError('Batched protocol verification failed')
except BaseException:
 import traceback
 error=traceback.format_exc();print(error,flush=True)
 (OUT/'throughput/batched_preflight_error.txt').write_text(error)
 raise
finally:
 try:
  if 'env' in globals():env.close()
 except Exception:pass
 app.close()
