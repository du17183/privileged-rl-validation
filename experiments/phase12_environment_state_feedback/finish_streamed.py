"""Overlap independent testing of finished runs with remaining training."""
import json,subprocess,time,traceback
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG

def run(module):
 subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m',f'experiments.{OUT.name}.{module}'],cwd=ROOT,check=True)
def main():
 with (LOG/'stream_heldout.log').open('x') as log:
  scheduler=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',f'experiments.{OUT.name}.stream_heldout'],
   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  while not ((OUT/'training_completed.json').exists() and (OUT/'heldout_completed.json').exists()):
   if list(OUT.glob('error_*')) or list(OUT.glob('heldout_error_*')) or (OUT/'heldout_scheduler_error.json').exists() or (OUT/'queue_continuation_error.json').exists():
    raise RuntimeError('Training or heldout failure recorded')
   code=scheduler.poll()
   if code is not None and (code or not (OUT/'heldout_completed.json').exists()):raise RuntimeError('Heldout scheduler failed')
   time.sleep(10)
  if scheduler.wait()!=0:raise RuntimeError('Heldout scheduler failed')
 run('verify_data');run('analyze');run('plot');run('report');run('verify_data')
 run('append_concurrency_report');run('append_stream_report')
 gate=json.loads((OUT/'extension_gate.json').read_text())
 if gate['passed']:
  (OUT/'extension_required.json').write_text(json.dumps(gate,indent=2));print('Matched 500k extension requires continuation',flush=True);return
 (OUT/'phase12_completed.json').write_text(json.dumps(dict(completed=True,completed_at=datetime.now(timezone.utc).isoformat(),
  training=json.loads((OUT/'training_completed.json').read_text()),heldout=json.loads((OUT/'heldout_completed.json').read_text()),
  extension=gate,report='docs/phase12_environment_state_feedback_report.md'),indent=2))
 run('bundle_results');print('Phase12 full pipeline completed',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (OUT/'postprocessing_error.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
