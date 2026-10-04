import os,subprocess
from experiments.phase12_environment_state_feedback.protocol import ROOT,LOG
with (LOG/'continue_queue.log').open('x') as f:
 proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.phase12_environment_state_feedback.continue_queue'],cwd=ROOT,
   env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('queue supervisor',proc.pid)
with (LOG/'throughput_benchmarks.log').open('x') as f:
 proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.phase12_environment_state_feedback.run_throughput_benchmarks'],cwd=ROOT,
   env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('evaluation benchmark',proc.pid)
