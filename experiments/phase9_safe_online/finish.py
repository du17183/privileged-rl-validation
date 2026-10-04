"""Complete bounded training, independent tests, statistics and report in order."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'

def run(script,*args):
    subprocess.check_call([sys.executable,f'experiments/phase9_safe_online/{script}',*args],cwd=ROOT)

def main():
    lock=(OUT/'finish.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run('run_heldout.py')
    while not (OUT/'training_completed.json').exists():
        if list(OUT.glob('error_*.txt')):
            raise RuntimeError('Training error artifact present; inspect before continuing')
        time.sleep(30)
    run('analyze.py')
    run('plot.py')
    run('baseline_inventory.py','--verify')
    run('finalize_report.py')
    (OUT/'phase9_completed.json').write_text(json.dumps(dict(training_runs=35,training_steps=10500000,
        heldout_verified=True,report='docs/phase9_safe_online_report.md')))
    print('Phase 9 completed and report written',flush=True)

if __name__=='__main__':main()
