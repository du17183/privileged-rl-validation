"""Export stable audited Phase15.3 evidence, not an Anchor or learning claim."""
from pathlib import Path
import hashlib,json,tarfile,time

R=Path.cwd();P=R/'phase15_3';O=P/'results';audit=json.loads((O/'final_audit.json').read_text())
assert audit['passed'] and audit['physical_controlled_comparisons_complete']
assert not audit['expert_usability']['gate_13_of16_each'], 'An expert prerequisite passed: finish the separately required data/action/learning branch before final delivery'
report=R/'docs/phase15_3_reacquire_report.md'
assert hashlib.sha256(report.read_bytes()).hexdigest()==audit['input_sha256'][str(report.relative_to(R))]
assert '尚不是最终算法验收报告' not in report.read_text(), 'Build the completed, scope-limited report before delivery'
tag=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());V=O/'delivery';V.mkdir(exist_ok=True)
files={report}
for p in P.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(P)
    if any(x in ['__pycache__','review_snapshots','delivery'] for x in rel.parts) or p.suffix in ['.pyc','.sock'] or p.name=='.analysis.lock':continue
    files.add(p)
snapshot={};stats={}
for p in sorted(files):
    s=p.stat();stats[p]=(s.st_size,s.st_mtime_ns)
    snapshot[str(p.relative_to(R))]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=s.st_size)
manifest=V/('manifest_'+tag+'.json')
manifest.write_text(json.dumps(dict(utc=tag,files=snapshot,file_count=len(snapshot),full_delivery_complete=True,
    meaning='All evidence and reports for the completed exploratory controlled diagnostic scope; does not mean severe recovery was solved',
    original_phase15_2_preserved=True,exploratory_model_seeds=1,expert_gate_passed=False,BC_pi05_training=False,
    GT_learning_input_ablation=False,severe_recovery_goal_achieved=False,new_anchor=False,phase16_allowed=False,
    omitted='Python caches, Unix socket endpoints, duplicate review archives and this export directory; historical inputs referenced by verified SHA',
    audit_sha256=hashlib.sha256((O/'final_audit.json').read_bytes()).hexdigest()),indent=2))
archive=V/('phase15_3_'+tag+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(files):
        s=p.stat();assert stats[p]==(s.st_size,s.st_mtime_ns),'Evidence changed during export '+str(p)
        t.add(p,arcname=str(p.relative_to(R)),recursive=False)
    t.add(manifest,arcname=str(manifest.relative_to(R)),recursive=False)
summary=dict(utc=tag,archive=str(archive.relative_to(R)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    manifest=str(manifest.relative_to(R)),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),file_count=len(snapshot),
    bytes_total=sum(x['bytes'] for x in snapshot.values()),local_SHA_verification_pending=True,full_delivery_complete=True,
    severe_recovery_goal_achieved=False,expert_gate_passed=False,new_anchor=False,phase16_allowed=False)
(V/'export_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
