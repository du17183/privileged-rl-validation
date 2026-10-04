"""Compact actual progress; validation is never labeled independent Final."""
import json,csv
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import OUT,CKPT,VARIANTS,run_name
def main():
 state=json.loads((OUT/'queue_state.json').read_text());runs=[];total=0
 for a,s in VARIANTS:
  for seed in range(5):
   name=run_name(a,s,seed);marker=CKPT/name/'completed.json';curve=OUT/f'eval_{name}_level2.csv'
   if marker.exists():
    m=json.loads(marker.read_text());steps=m['steps'];status='completed'
   elif curve.exists():
    with curve.open() as f:rows=list(csv.DictReader(f))
    steps=int(rows[-1]['env_steps']) if rows else 0;status='running'
    m={}
   else:continue
   total+=steps
   runs.append(dict(name=name,status=status,recorded_train_steps=steps,
       online_successes=m.get('online_successes'),online_episodes=m.get('online_episodes')))
 result=dict(timestamp=datetime.now(timezone.utc).isoformat(),last_queue_timestamp=state['timestamp'],
   running=len(state['running']),pending=len(state['pending']),completed=len(state['completed']),
   blocked_by_external_compute_gpus=state.get('blocked_by_external_compute_gpus'),
   recorded_training_steps=total,target_training_steps=15000000,runs=runs,
   independent_heldout_completed=(OUT/'heldout_completed.json').exists(),
   finished=(OUT/'phase12_completed.json').exists(),extension_required=(OUT/'extension_required.json').exists(),
   errors=[p.name for pattern in ('error_*','heldout_error_*','postprocessing_error.json') for p in OUT.glob(pattern)])
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
