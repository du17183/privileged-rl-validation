"""Read-only identity/input monitor; output confined to Phase15.3."""
from pathlib import Path
import json,hashlib,time,os
R=Path.cwd().resolve();A=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
issues=[];checked={};active=[];exited=[];reused=[]
for p,v in A['sha256'].items():
 if p.endswith('.py') or p.endswith('.pt') or p in ['datasets/random_door_expert/collection_v1/trajectories.h5','datasets/random_door_expert/split_v1/split.json']:
  x=sha(R/p);checked[p]=x
  if x!=v:issues.append(['frozen_input_drift',p,v,x])
for p in (R/'phase15_3/results').glob('*launch*.json'):
 x=json.loads(p.read_text());q=Path('/proc')/str(x['pid'])/'cmdline'
 if q.exists() and q.read_bytes():
  argv=[v.decode() for v in q.read_bytes().split(b'\0') if v]
  if argv==x['command']:active.append(dict(role='supervisor',pid=x['pid'],argv=argv))
  else:reused.append(dict(pid=x['pid'],actual=argv,expected=x['command']))
 else:exited.append(dict(role='supervisor',pid=x['pid']))
for p in (R/'phase15_3/experiments').rglob('lifecycle.json'):
 for x in json.loads(p.read_text()).get('processes',[]):
  pr=Path('/proc')/str(x['pid']);q=pr/'cmdline'
  if not q.exists() or not q.read_bytes():exited.append(dict(pid=x['pid'],role=x['role'],exit_code=x.get('exit_code')));continue
  argv=[v.decode() for v in q.read_bytes().split(b'\0') if v]
  if argv!=x['command']:reused.append(dict(pid=x['pid'],actual=argv,expected=x['command']));continue
  status=(pr/'status').read_text();ppid=int(next(v.split()[1] for v in status.splitlines() if v.startswith('PPid:')))
  if ppid!=x['ppid']:issues.append(['active_worker_ppid',x['pid'],ppid,x['ppid']])
  active.append(dict(pid=x['pid'],role=x['role'],ppid=ppid,argv=argv,gpu=x['gpu']))
assert len([x for x in active if x['role']=='evaluation'])<=2
new={str(p.relative_to(R)):sha(p) for p in (R/'phase15_3').rglob('*.py')}
report=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),passed=not issues,issues=issues,frozen_sha256=checked,new_source_sha256=new,
 active=active,exited=exited,reused_pids=reused,scope='read-only source/weight/identity checks; no process or worker mutation',training=False)
out=R/'phase15_3/results'/('input_identity_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json');out.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=report['passed'],issues=issues,active=active,reused_pids=reused,checked=len(checked),snapshot=str(out.relative_to(R))),indent=2))
assert not issues
