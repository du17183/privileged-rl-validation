"""Complete the bounded experiment pipeline; gate reporting on real results."""
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'


def main():
    while not (OUT/'training_completed.json').exists():
        errors=list(OUT.glob('error_*.txt'))
        if errors:raise RuntimeError(f'Training errors: {errors}')
        time.sleep(15)
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','experiments.phase10_quality_lwd.run_heldout'],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','experiments.phase10_quality_lwd.preservation_supplement','--verify'],cwd=ROOT,check=True)
    for module in ('baseline_inventory','analyze','plot','finalize_report'):
        command=[str(ROOT/'.venv/bin/python'),'-m',f'experiments.phase10_quality_lwd.{module}']
        if module=='baseline_inventory':command.append('--verify')
        subprocess.run(command,cwd=ROOT,check=True)
    report=ROOT/'docs/phase10_quality_lwd_report.md'
    (OUT/'phase10_completed.json').write_text(json.dumps(dict(completed_at=datetime.now(timezone.utc).isoformat(),
        runs=35,training_steps=10500000,independent_tests_complete=True,report=str(report),
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()),indent=2))
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','experiments.phase10_quality_lwd.bundle_results'],cwd=ROOT,check=True)
    print('Phase10 actual matched training, independent tests, statistics and final report complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (OUT/'pipeline_error.txt').write_text(traceback.format_exc())
        raise
