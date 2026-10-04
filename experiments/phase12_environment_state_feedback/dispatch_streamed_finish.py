"""Replace only the waiting postprocessing supervisor, before any heldout has started."""
import hashlib,json,os,subprocess
import psutil
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG

if (OUT/'training_completed.json').exists() or (OUT/'heldout_status.csv').exists():raise RuntimeError('Old postprocessing may be active')
original=json.loads((OUT/'training_source_manifest.json').read_text())
amendment=json.loads((OUT/'throughput_amendment.json').read_text())
for manifest in (original,amendment['source_manifest']):
 changes=[p for p,h in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
 if changes:raise RuntimeError(f'Immutable source changed: {changes}')
proc=psutil.Process(2960989)
if proc.cmdline()[-2:]!=['-m','experiments.phase12_environment_state_feedback.finish_fast']:raise RuntimeError('Unexpected supervisor identity')
if proc.children(recursive=True):raise RuntimeError('Old supervisor has active children')
names=('stream_heldout.py','finish_streamed.py','append_stream_report.py','dispatch_streamed_finish.py')
manifest={str((ROOT/'experiments'/OUT.name/n).relative_to(ROOT)):hashlib.sha256((ROOT/'experiments'/OUT.name/n).read_bytes()).hexdigest() for n in names}
record=dict(created_at=datetime.now(timezone.utc).isoformat(),reason='User requested higher throughput; four GPUs idle while waiting for tail training',
 source_manifest=manifest,evaluation='Original heldout and metrics unchanged; complete-marker readiness, exclusive GPU admission, cap6 independent processes/GPU',
 initial_training_queue=json.loads((OUT/'queue_state.json').read_text()),old_sources_unchanged=True)
with (OUT/'streaming_evaluation_amendment.json').open('x') as f:json.dump(record,f,indent=2)
proc.terminate();proc.wait(timeout=15)
with (LOG/'finish_streamed.log').open('x') as f:
 worker=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',f'experiments.{OUT.name}.finish_streamed'],cwd=ROOT,
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(worker.pid)
