import os,subprocess,json
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG
for count in (32,64,128,256):
 with (LOG/f'throughput_batch_{count}.log').open('x') as f:
  subprocess.run([str(ROOT/'.venv/bin/python'),'-u',f'experiments/{OUT.name}/throughput_benchmark.py','--count',str(count),'--device','cuda:0'],cwd=ROOT,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(json.loads((OUT/'throughput'/f'batch_{count}.json').read_text()),flush=True)
