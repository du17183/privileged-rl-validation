"""Retrospective natural-stage statistics; never used to select a repair."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
from scipy.stats import t
R=Path.cwd();P=R/'phase15_2/results';O=R/'phase15_3/results/retrospective';O.mkdir(parents=True,exist_ok=True)
SOURCE=P/'metrics/funnel_v2_episode.csv'
rows=list(csv.DictReader(SOURCE.open()))
stages=['source','trigger_proxy','retreat_candidate','safe_pose','pregrasp_plan','pregrasp_arrival','reapproach','recontact','valid_grasp','handoff','normal_takeover','door_progress','final_success']
def flag(x):return {'True':True,'False':False,'unknown':None}.get(x)
def write(name,records):
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def estimate(v):
    a=np.array([x for x in v if x is not None]);n=len(a)
    if not n:return dict(mean=None,sd=None,ci_low=None,ci_high=None,seed_count=0)
    mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else None
    delta=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    return dict(mean=mean,sd=sd,ci_low=mean-delta if delta is not None else None,ci_high=mean+delta if delta is not None else None,seed_count=n)
per=[];ag=[];fail=[]
for method in 'ABCDE':
 for population in ['natural','graded']:
  for bucket in [-1,0,1,2,3,4]:
   selected=[r for r in rows if r['method']==method and r['population']==population and (bucket==-1 or int(r['distance_bin'])==bucket)]
   for mode in ['raw','ordered']:
    prefix='' if mode=='raw' else 'ordered_'
    for k,stage in enumerate(stages):
     ss=[]
     for seed in range(5):
      rr=[r for r in selected if int(r['seed'])==seed];prev=stages[k-1] if k else 'source'
      entered=[r for r in rr if flag(r[prefix+prev]) is True]
      success=sum(flag(r[prefix+stage]) is True for r in entered)
      known=sum(flag(r[prefix+stage]) is not None for r in entered)
      coverage=sum(flag(r[prefix+stage]) is True for r in rr)
      d=dict(method=method,population=population,distance_bin=bucket,mode=mode,seed=seed,stage=stage,
       source_n=len(rr),entered_n=len(entered),success_n=success,unknown_n=len(entered)-known,
       coverage_n=coverage,conditional_rate=success/known if known else None,cumulative_rate=coverage/len(rr) if rr else None)
      per.append(d);ss.append(d)
     d=dict(method=method,population=population,distance_bin=bucket,mode=mode,stage=stage,
       source_n=sum(s['source_n'] for s in ss),entered_n=sum(s['entered_n'] for s in ss),
       success_n=sum(s['success_n'] for s in ss),unknown_n=sum(s['unknown_n'] for s in ss),coverage_n=sum(s['coverage_n'] for s in ss))
     for key in ['conditional_rate','cumulative_rate']:
      d.update({key+'_'+a:b for a,b in estimate([s[key] for s in ss]).items()})
     known=d['entered_n']-d['unknown_n']
     d['conditional_rate_pooled']=d['success_n']/known if known else None
     d['cumulative_rate_pooled']=d['coverage_n']/d['source_n'] if d['source_n'] else None
     ag.append(d)
   for seed in range(5):
    rr=[r for r in selected if int(r['seed'])==seed];ff=[r for r in rr if r['primary']!='success']
    for category in ['F'+str(i) for i in range(1,16)]:
     n=sum(r['primary']==category for r in ff)
     fail.append(dict(method=method,population=population,distance_bin=bucket,seed=seed,category=category,count=n,failure_n=len(ff),share=n/len(ff) if ff else None))
    assert sum(sum(r['primary']=='F'+str(i) for r in ff) for i in range(1,16))==len(ff)
write('stage_per_seed.csv',per);write('stage_aggregate.csv',ag);write('exclusive_failures.csv',fail)
natural=[r for r in rows if r['population']=='natural'];assert len(natural)==2500
assert len({(r['method'],r['seed'],r['cohort_id']) for r in natural})==2500
e=[r for r in natural if r['method']=='E'];assert sum(flag(r['final_success']) is True for r in e)==14
losses=[]
for a,b in [('retreat_candidate','safe_pose'),('pregrasp_plan','pregrasp_arrival'),('valid_grasp','handoff'),('door_progress','final_success')]:
 losses.append(dict(previous=a,next=b,previous_n=sum(flag(r[a]) is True for r in e),next_n=sum(flag(r[b]) is True for r in e),loss_n=sum(flag(r[a]) is True and flag(r[b]) is False for r in e)))
manifest=dict(source=str(SOURCE.relative_to(R)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),episodes=len(rows),natural_episodes=2500,graded_episodes=10000,
    definitions='entered = preceding observed milestone; success = both observed; coverage independently observed current milestone. Raw across attempts/bypasses is not a single contiguous causal path. Ordered uses original all13 milestones intersection. Trigger is the saved threshold proxy, not an intervention-defined false-trigger classifier.',
    distance_bins_cm=['[0,5)','[5,10)','[10,15)','[15,20)','[20,+inf)'],
    natural_low_distance_missing=True,actual_collision='unknown',five_seed_exact_min=.0625,E_raw_set_difference_losses=losses,selection=False,training=False)
(O/'scope.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
