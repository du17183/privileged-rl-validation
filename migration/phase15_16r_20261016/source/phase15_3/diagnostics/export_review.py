"""Stable review snapshot. Never calls a worker or labels delivery complete."""
from pathlib import Path
import hashlib,json,tarfile,time
R=Path.cwd();tag=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());V=R/'phase15_3/results/review_snapshots';V.mkdir(exist_ok=True)
files={R/'docs/phase15_3_reacquire_report.md',R/'phase15_3/results/phase15_2_baseline_audit.json'}
for p in (R/'phase15_3').rglob('*.py'):files.add(p)
files.update((R/'phase15_3/experiments').glob('*.json'))
for dirname in ['phase15_3/results/retrospective','phase15_3/results/controlled_analysis','phase15_3/results/figures',
                'phase15_3/results/candidate_contract_gate','phase15_3/diagnostics/cohorts/development',
                'phase15_3/experiments/frozen600_seed1_first8_attempt003','phase15_3/experiments/new_natural_development_seed1']:
    files.update(p for p in (R/dirname).rglob('*') if p.is_file())
time_root=R/'phase15_3/experiments/time600_900_1200_seed1_first8_v1'
for d in time_root.glob('budget*'):
    if (d/'complete.json').exists():files.update(p for p in d.rglob('*') if p.is_file())
ctrl=R/'phase15_3/experiments/controlled_dev16_seed1_v1'
analysis=R/'phase15_3/results/controlled_analysis/analysis_scope.json'
if analysis.exists():
    for name in json.loads(analysis.read_text())['closed_cases']:
        if name in ['frozen_E','GT_direct','GT_projected_retreat','GT_pose_pregrasp']:
            files.update(p for p in (ctrl/name/'results').rglob('*') if p.is_file())
# Inputs are locked. Active logs, lifecycle and shared motion stream are omitted.
for root in [ctrl,time_root]:
    for name in ['scope.json','jobs.json']:
        if (root/name).exists():files.add(root/name)
for p in (R/'phase15_3/results').glob('input_identity_*.json'):files.add(p)
snapshot={};before={}
for p in sorted(files):
    rel=str(p.relative_to(R));assert '..' not in Path(rel).parts
    s=p.stat();before[p]=(s.st_size,s.st_mtime_ns)
    snapshot[rel]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=s.st_size)
manifest=V/('review_'+tag+'.json')
manifest.write_text(json.dumps(dict(utc=tag,files=snapshot,file_count=len(snapshot),full_delivery_complete=False,
    controlled_pending=not(ctrl/'complete.json').exists(),time_pending=not(time_root/'complete.json').exists(),
    scope='stable completed evidence and current working report; excludes active output streams; review only',training=False),indent=2))
tar=V/('review_'+tag+'.tar.gz')
with tarfile.open(tar,'w:gz') as t:
    for p in sorted(files):
        s=p.stat();assert before[p]==(s.st_size,s.st_mtime_ns)
        t.add(p,arcname=str(p.relative_to(R)),recursive=False)
    t.add(manifest,arcname=str(manifest.relative_to(R)),recursive=False)
print(json.dumps(dict(archive=str(tar.relative_to(R)),archive_sha256=hashlib.sha256(tar.read_bytes()).hexdigest(),manifest=str(manifest.relative_to(R)),file_count=len(snapshot),full_delivery_complete=False)))
