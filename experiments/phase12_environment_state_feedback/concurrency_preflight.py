"""Bounded throughput test of four independent original 32-environment evaluators."""
import json,os,subprocess,time
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG

def main():
 jobs=[];start=time.perf_counter()
 for i in range(4):
  label=f'concurrency4_worker{i}'
  log=(LOG/f'{label}.log').open('x')
  p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',f'experiments/{OUT.name}/throughput_benchmark.py',
   '--count','32','--label',label,'--device','cuda:0'],cwd=ROOT,env=dict(os.environ,
   CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
  jobs.append((p,log,label))
 results=[]
 for p,log,label in jobs:
  code=p.wait();log.close()
  if code:raise RuntimeError(f'Owned concurrency benchmark {label} failed: {code}')
  results.append(json.loads((OUT/'throughput'/f'{label}.json').read_text()))
 wall=time.perf_counter()-start
 # Startup/teardown are included in orchestration wall, excluded from active-loop comparison.
 interactions=sum(r['metrics']['eval_env_steps'] for r in results)
 active=max(r['seconds'] for r in results)
 reference=json.loads((OUT/'throughput/batch_32.json').read_text())
 result=dict(workers=4,parallel_envs_per_worker=32,physical_interactions=interactions,
  active_seconds=active,wall_including_startup=wall,aggregate_env_steps_per_second=interactions/active,
  wall_env_steps_per_second=interactions/wall,reference_env_steps_per_second=reference['env_steps_per_second'],
  active_throughput_ratio=(interactions/active)/reference['env_steps_per_second'],
  caveat='Concurrent background training and warmup differ; this is a bounded operational estimate, not a policy comparison.',
  passed=all(r['metrics']['episodes']==32 for r in results) and interactions/active>1.2*reference['env_steps_per_second'])
 (OUT/'throughput/concurrency_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
