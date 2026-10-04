"""Replace only the idle postprocessing supervisor before heldout starts."""
import hashlib,json,os,subprocess
import psutil
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG
gate=json.loads((OUT/'throughput/batched_preflight.json').read_text())
if not gate['passed']:raise RuntimeError('Batched preflight not passed')
if (OUT/'training_completed.json').exists() or (OUT/'heldout_status.csv').exists():raise RuntimeError('Existing postprocessing may be active')
original=json.loads((OUT/'training_source_manifest.json').read_text())
changes=[p for p,h in original.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
if changes:raise RuntimeError(f'Frozen code changed: {changes}')
proc=psutil.Process(1076334)
if proc.cmdline()[-2:]!=['-m','experiments.phase12_environment_state_feedback.finish']:raise RuntimeError('Unexpected supervisor')
manifest={}
for name in ('continue_queue.py','throughput_benchmark.py','run_throughput_benchmarks.py','batched_evaluation.py',
 'heldout_batched.py','batched_preflight.py','run_heldout_fast.py','finish_fast.py','append_throughput_report.py'):
 p=ROOT/'experiments'/OUT.name/name;manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(approved_scope='User requested higher throughput; preserve all methods, seeds, physical reset plans, episode budgets and frozen learning code',
 created_at=datetime.now(timezone.utc).isoformat(),training_concurrency='Last two adopted safely; max4/GPU, all50 launched',
 evaluation='8 independent original32-clone test groups per process; max6 processes/GPU',source_manifest=manifest,
 passed_preflight=gate,original_frozen_source_changes=[])
with (OUT/'throughput_amendment.json').open('x') as f:json.dump(record,f,indent=2)
proc.terminate();proc.wait(timeout=15)
with (LOG/'finish_fast.log').open('x') as f:
 worker=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.phase12_environment_state_feedback.finish_fast'],cwd=ROOT,
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(worker.pid)
