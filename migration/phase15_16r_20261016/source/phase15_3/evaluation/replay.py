"""Frozen600 E development replay, with directly paired physical motion records."""
import ast,hashlib,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
R=Path.cwd();B=Path(os.environ['PHASE15_3_OUTPUT']);B.mkdir(parents=True,exist_ok=True)
audit=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text());assert audit['passed']
source=R/'phase15_2/evaluation/full_loop_eval.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['sha256'][str(source.relative_to(R))]
tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))]
ns=dict(__name__='phase15_3_frozen_development_probe',__file__=str(source));exec(compile(tree,str(source),'exec'),ns)
import phase15_2.common as common
import phase15_1.severe_reacquire.pregrasp_planner as old
common.RESULTS=B;old.RESULTS=B
Policy=ns['ChainPolicy'];of=Policy.forward;oo=Policy.observe
stream=(B/'motion_steps.jsonl').open('x');pending=None;calls=0;started=time.monotonic()
def native(v):
    import numpy as np
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def forward(self,x):
    global pending,calls
    action=of(self,x);calls+=1
    import numpy as np
    records=[];r=self.before_action
    for i in np.flatnonzero(self.active):
        p=self.planner;meta=p.current[i] if p else None
        target=None;cursor=None
        if p and p.path[i] is not None:
            cursor=int(p.cursor[i]);path=p.path[i]
            if cursor<len(path):target=path[cursor].copy()
        records.append(dict(clone=int(i),batch=int(self.batch),episode=int(self.episode[i]),tick=int(round(r[i,25]*600)),
            router=int(self.router.state[i]),phase=str(p.phase[i]) if p else None,
            q_pre=r[i,:9].copy(),qd_pre=r[i,9:18].copy(),eef_pre=r[i,18:25].copy(),contact_pre=r[i,30:32].copy(),
            handle_pre=r[i,32:39].copy(),action=action[i].detach().cpu().numpy(),path_cursor=cursor,
            target_q=target,planner_elapsed=int(p.elapsed[i]) if p else None,
            failure_stage=(meta or {}).get('failure_stage'),trigger_tick=(meta or {}).get('trigger_tick')))
    pending=records
    return action
def observe(self,robot,gt,done):
    global pending
    if pending is not None:
        r=robot.detach().cpu().numpy();g=gt.detach().cpu().numpy();d=done.detach().cpu().numpy()
        for row in pending:
            i=row['clone'];row.update(q_post=r[i,:9],qd_post=r[i,9:18],eef_post=r[i,18:25],contact_post=g[i,9:11],door_post=float(g[i,0]),terminal=bool(d[i]),state_timing='pre issued action; post captured transition_before_auto_reset')
            stream.write(json.dumps(row,default=native)+'\n')
        stream.flush();pending=None
    return oo(self,robot,gt,done)
Policy.forward=forward;Policy.observe=observe
code=0;error=None
try:ns['main']()
except BaseException:
    code=1;error=traceback.format_exc();print(error,flush=True)
finally:
    stream.close()
    for p in (B/'results').glob('natural_severe.json'):
        data=json.loads(p.read_text());data.update(phase15_3=True,diagnostic_only=True,formal_statistics=False,
            exploratory_model_seeds=1,source_split='reused Phase15.2 validation',algorithm_changed=False,instrumentation_overhead='unknown, no paired uninstrumented closed-loop run')
        p.write_text(json.dumps(data,indent=2))
    (B/'exit.json').write_text(json.dumps(dict(exit_code=code,error=error,elapsed_s=time.monotonic()-started,forward_calls=calls,training=False),indent=2))
    ns['app'].close()
sys.exit(code)
