"""Detach one explicitly named Phase12 process with its own log."""
import argparse
import os
import subprocess
from experiments.phase12_environment_state_feedback.protocol import ROOT, LOG
p = argparse.ArgumentParser();p.add_argument('module', choices=('launch', 'finish'));args = p.parse_args()
path = LOG/f'{args.module}.log'
with path.open('x') as log:
    proc = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', '-m', f'experiments.phase12_environment_state_feedback.{args.module}'],
        cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(proc.pid)
