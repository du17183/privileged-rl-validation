"""Complete already authorized evaluation/analysis after bounded training."""
import json
import os
import subprocess
import time
import traceback
from datetime import datetime,timezone
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT


def run(module,*args):
    subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m',f'experiments.phase11_parameter_generalization.{module}',*args],cwd=ROOT,check=True)


def main():
    while not (OUT/'training_completed.json').exists():
        if list(OUT.glob('error_*')): raise RuntimeError('Training error recorded')
        time.sleep(10)
    run('verify_data')
    if not (OUT/'heldout_completed.json').exists(): run('run_heldout')
    run('analyze')
    gate=json.loads((OUT/'conditional_gate.json').read_text())
    if gate['passed'] and not (OUT/'conditional_completed.json').exists():
        run('run_heldout','--conditional')
    else:
        (OUT/'conditional_deferred.json').write_text(json.dumps(gate,indent=2))
    if not (OUT/'parameter_diagnosis.json').exists(): run('diagnose_runner')
    run('plot')
    run('finalize_report')
    run('verify_data')
    (OUT/'phase11_completed.json').write_text(json.dumps(dict(completed=True,completed_at=datetime.now(timezone.utc).isoformat(),
        training=json.loads((OUT/'training_completed.json').read_text()),heldout=json.loads((OUT/'heldout_completed.json').read_text()),
        conditional=gate,report='docs/phase11_parameter_generalization_report.md'),indent=2))
    run('bundle_results')
    print('Phase11 full pipeline completed',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'postprocessing_error.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
