"""Resume postprocessing only; preserve the first failure and raw experiments."""
import json
import os
import subprocess
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, LOG
error=OUT/'postprocessing_error.json'
if error.exists(): error.rename(OUT/'postprocessing_first_failure.json')
with (LOG/'finish_resumed.log').open('x') as log:
    proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.phase11_parameter_generalization.finish'],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(proc.pid)
