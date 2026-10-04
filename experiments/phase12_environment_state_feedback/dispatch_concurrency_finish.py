"""Replace only the idle postprocessing supervisor; original evaluator is unchanged."""
import hashlib,json,os,subprocess
import psutil
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG

gate=json.loads((OUT/'throughput/concurrency_preflight.json').read_text())
if not gate['passed']:raise RuntimeError('Independent evaluator concurrency preflight failed')
if (OUT/'training_completed.json').exists() or (OUT/'heldout_status.csv').exists():raise RuntimeError('Existing postprocessing may be active')
original=json.loads((OUT/'training_source_manifest.json').read_text())
changes=[p for p,h in original.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
if changes:raise RuntimeError(f'Frozen code changed: {changes}')
proc=psutil.Process(1076334)
if proc.cmdline()[-2:]!=['-m','experiments.phase12_environment_state_feedback.finish']:raise RuntimeError('Unexpected supervisor')
if proc.children(recursive=True):raise RuntimeError('Old supervisor has active children')
names=('continue_queue.py','throughput_benchmark.py','run_throughput_benchmarks.py','concurrency_preflight.py',
 'run_heldout_fast.py','finish_fast.py','append_concurrency_report.py','dispatch_concurrency_finish.py')
manifest={str((ROOT/'experiments'/OUT.name/n).relative_to(ROOT)):hashlib.sha256((ROOT/'experiments'/OUT.name/n).read_bytes()).hexdigest() for n in names}
record=dict(approved_scope='User requested higher throughput; preserve frozen training and evaluation methods, seeds, physical plans, episode budgets',
 created_at=datetime.now(timezone.utc).isoformat(),training_concurrency='Last two adopted safely; max4/GPU, all50 launched',
 evaluation='Original heldout.py and metrics.py unchanged; original32 environments/process, max6 independent processes/GPU',
 source_manifest=manifest,passed_preflight=gate,rejected_batched_evaluation=json.loads((OUT/'throughput/batched_preflight.json').read_text()),
 original_frozen_source_changes=[])
with (OUT/'throughput_amendment.json').open('x') as f:json.dump(record,f,indent=2)
proc.terminate();proc.wait(timeout=15)
with (LOG/'finish_fast.log').open('x') as f:
 worker=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.phase12_environment_state_feedback.finish_fast'],cwd=ROOT,
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(worker.pid)
