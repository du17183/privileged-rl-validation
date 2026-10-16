"""Exact initial-array and action/physics trajectory comparisons, closed cases."""
from pathlib import Path
import json,hashlib,time
import numpy as np

R=Path.cwd();A=R/'phase15_3/results/controlled_analysis';O=R/'phase15_3/results/trajectory_audit';O.mkdir(exist_ok=True)
s=json.loads((A/'analysis_scope.json').read_text());present=set(s['closed_cases']);out=[]
def root(case):
    E=R/'phase15_3/experiments'
    if case=='probe600':return E/'frozen600_seed1_first8_attempt003/results'
    if case.startswith('time'):return E/f'time600_900_1200_seed1_first8_v1/budget{case[4:]}/results'
    if case.startswith('confirm_'):return E/'confirmation_seed1_v1'/case[8:]/'results'
    return E/'controlled_dev16_seed1_v1'/case/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pairs=[(x['baseline'],x['candidate']) for x in json.loads((A/'initial_pairing.json').read_text())['pairs']]
fields=['action','q_pre','qd_pre','q_post','qd_post','eef_pre','eef_post','contact_pre','contact_post','door_post']
for left,right in pairs:
    if left not in present or right not in present:continue
    za=np.load(root(left)/'natural_severe.initial.npz');zb=np.load(root(right)/'natural_severe.initial.npz');ids=list(za['ids']);order=[list(zb['ids']).index(x) for x in ids]
    initial={}
    for key in ['robot','state']:
        a=za[key];b=zb[key][order];initial[key]=dict(dtype_equal=a.dtype==b.dtype,numeric_exact=bool(np.array_equal(a,b)),
            bytes_exact=np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes(),shape=list(a.shape))
    paths=[A/(c+'_closed_motion.jsonl') for c in [left,right]]
    def load(p):return {(x['clone'],x['episode'],x['tick']):x for line in p.read_text().splitlines() for x in [json.loads(line)]}
    ma,mb=[load(p) for p in paths];is_time=left=='probe600' or left.startswith('time')
    if is_time:ma={k:v for k,v in ma.items() if k[2]<600};mb={k:v for k,v in mb.items() if k[2]<600}
    common=set(ma)&set(mb);diff={};unequal={}
    for f in fields:
        d=[float(np.max(np.abs(np.asarray(ma[k][f])-np.asarray(mb[k][f])))) for k in common]
        diff[f]=max(d) if d else None;unequal[f]=sum(v!=0 for v in d)
    first=min((k[2] for k in common if any(not np.array_equal(ma[k][f],mb[k][f]) for f in fields)),default=None)
    out.append(dict(baseline=left,candidate=right,initial=initial,key_sets_equal=set(ma)==set(mb),shared_records=len(common),
        baseline_records=len(ma),candidate_records=len(mb),maximum_numeric_difference=diff,unequal_records_by_field=unequal,
        first_observed_divergence_tick=first,compared_window='before600 ticks' if is_time else 'entire completed motion stream',
        motion_sha256={str(p.relative_to(R)):sha(p) for p in paths},
        interpretation='Equal observed trajectories mean this intervention did not change measured behavior in this cohort; not proof of equivalence outside it'))
result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),comparisons=out,
    all_initial_arrays_byte_exact=all(v['dtype_equal'] and v['bytes_exact'] for x in out for v in x['initial'].values()),
    source_sha256=sha(Path(__file__)),model_seeds=1,contact_history_limitation='Saved prefix and observed initial contact matched; hidden PhysX cache was not directly read')
(O/'scope.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
print(json.dumps([{k:x[k] for k in ['baseline','candidate','shared_records','first_observed_divergence_tick','compared_window']} for x in out],indent=2))
