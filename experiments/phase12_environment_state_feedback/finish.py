"""Independent evaluation and analysis after all bounded training is complete."""
import json,subprocess,time,traceback
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT
def run(module):
 subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m',f'experiments.{OUT.name}.{module}'],cwd=ROOT,check=True)
def main():
 while not (OUT/'training_completed.json').exists():
  if list(OUT.glob('error_*')):raise RuntimeError('Training error recorded')
  time.sleep(10)
 run('verify_data')
 if not (OUT/'heldout_completed.json').exists():run('run_heldout')
 run('analyze');run('plot');run('report');run('verify_data')
 gate=json.loads((OUT/'extension_gate.json').read_text())
 if gate['passed']:
  (OUT/'extension_required.json').write_text(json.dumps(gate,indent=2));print('300k results completed; qualifying matched extension requires continuation',flush=True);return
 (OUT/'phase12_completed.json').write_text(json.dumps(dict(completed=True,completed_at=datetime.now(timezone.utc).isoformat(),
    training=json.loads((OUT/'training_completed.json').read_text()),heldout=json.loads((OUT/'heldout_completed.json').read_text()),
    extension=gate,report='docs/phase12_environment_state_feedback_report.md'),indent=2))
 run('bundle_results');print('Phase12 full pipeline completed',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (OUT/'postprocessing_error.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
