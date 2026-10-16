"""Independent input/endpoint audit after reports, without extra simulation."""
import csv
import numpy as np
import h5py
from scipy import stats
from phase15_2.common import ROOT,RESULTS,read,save_json,sha256
from phase15_2.run import MAIN,GRADES


def equal_number(actual,expected,name):
    if actual is None or expected is None:
        if actual is not None or expected is not None:raise RuntimeError('Null statistic mismatch '+name)
    elif not np.isclose(float(actual),float(expected),atol=1e-10,rtol=1e-10,equal_nan=True):
        raise RuntimeError('Independent statistic mismatch '+name)


def independent_interval(values):
    values=np.asarray(values,float)
    if values.shape!=(5,):raise RuntimeError('Expected five paired model seeds')
    mean=float(np.sum(values)/5);sd=float(np.sqrt(np.sum((values-mean)**2)/4))
    half=2.7764451051977987*sd/np.sqrt(5)
    return dict(mean=mean,low=mean-half,high=mean+half,sd=sd)


def independent_statistics(means,summary):
    art=['ee_offset','contact_loss','door_regression','stagnation']
    values=dict(means)
    for method in ['A','B','C','D','E']:
        values[method,'artificial_macro']=np.sum([means[method,test] for test in art],axis=0)/4
    expected_keys=set(values)
    if {(row['method'],row['test']) for row in summary['aggregates']}!=expected_keys:
        raise RuntimeError('Aggregate endpoint coverage')
    for row in summary['aggregates']:
        expected=independent_interval(values[row['method'],row['test']])
        for field,value in expected.items():equal_number(row[field],value,str((row['method'],row['test'],field)))
        if row['seeds']!=5:raise RuntimeError('Seed count mismatch')
    comparisons={};tests=['fixed','random','artificial_macro','natural_severe','full_loop']
    for left,right in [('B','A'),('C','B'),('D','C'),('E','D'),('E','A')]:
        for test in tests:
            delta=np.asarray(values[left,test])-np.asarray(values[right,test])
            expected=independent_interval(delta)
            threshold=abs(float(np.sum(delta)/5))-1e-12
            exceed=0
            for mask in range(32):
                statistic=abs(sum(delta[i]*(1 if mask&(1<<i) else -1) for i in range(5))/5)
                exceed+=statistic>=threshold
            expected['exact_signflip_p']=exceed/32
            if np.std(delta)>0:
                t=float(np.mean(delta)/(np.std(delta,ddof=1)/np.sqrt(5)))
                expected['t_test_p']=float(2*stats.t.sf(abs(t),4))
            else:expected['t_test_p']=None
            comparisons[left,right,test]=expected
    ordered=sorted(comparisons,key=lambda key:comparisons[key]['exact_signflip_p']);previous=0
    for rank,key in enumerate(ordered):
        previous=max(previous,min(1,(25-rank)*comparisons[key]['exact_signflip_p']))
        comparisons[key]['holm_exact_p']=previous
    if {(r['left'],r['right'],r['test']) for r in summary['comparisons']}!=set(comparisons):
        raise RuntimeError('Paired test family coverage')
    for row in summary['comparisons']:
        key=row['left'],row['right'],row['test']
        for field,value in comparisons[key].items():equal_number(row[field],value,str(key+(field,)))
    for test,threshold in [('fixed',.95),('random',.90),('artificial_macro',.85),('natural_severe',.50),('full_loop',.90)]:
        row=summary['gates'][test];expected=independent_interval(values['E',test])
        equal_number(row['threshold'],threshold,test+' threshold')
        equal_number(row['mean'],expected['mean'],test+' gate mean')
        equal_number(row['ci_low'],expected['low'],test+' gate CI')
        mean_pass=expected['mean']>threshold if test=='natural_severe' else expected['mean']>=threshold
        ci_pass=expected['low']>threshold if test=='natural_severe' else expected['low']>=threshold
        if row['mean_pass']!=mean_pass or row['lower_ci_pass']!=ci_pass:raise RuntimeError('Gate truth mismatch')
    return len(expected_keys),len(comparisons)


def main():
    counts={};means={};trials=0;interactions=0;art=['ee_offset','contact_loss','door_regression','stagnation']
    summary=read(RESULTS/'summary.json')
    for source,digest in read(RESULTS/'frozen_inputs.json')['files'].items():
        if sha256(ROOT/source)!=digest:raise RuntimeError('Frozen history modified '+source)
    for method in ['A','B','C','D','E']:
        for seed in range(5):
            for test in MAIN+GRADES:
                path=RESULTS/f'test/{method}_seed{seed}{"_grades" if test in GRADES else ""}/{test}.json'
                d=read(path);rows=read(path.with_suffix('.telemetry.json'))
                if len(rows)!=d['episodes']:raise RuntimeError('Episode coverage missing')
                identities=[r.get('cohort_id',(r['clone'],r['episode'])) for r in rows]
                if len(set(identities))!=len(rows):raise RuntimeError('Duplicate terminal')
                if any(r['original_episode_ticks']>600 for r in rows):raise RuntimeError('600tick budget exceeded')
                if abs(sum(r['success'] for r in rows)/len(rows)-d['success'])>1e-10:raise RuntimeError('Metric mismatch')
                expected_n=64 if test=='fixed' else 128 if test in ['random','full_loop'] else 100
                if d['episodes']!=expected_n:raise RuntimeError('Formal endpoint quota mismatch')
                if d['rl_updates']!=0:raise RuntimeError('Unexpected training')
                if method in ['C','D','E'] and d['chain_config'].get('entry_distance_exit_m')!=.06:
                    raise RuntimeError('Stale absolute entry criterion in formal output')
                trials+=len(rows);interactions+=d['interactions']
                with h5py.File(path.with_suffix('.stage_trace.h5'),'r') as trace:
                    data=trace['steps'][:]
                    if int(data['terminal'].sum())<expected_n:raise RuntimeError('Trace terminal coverage')
                    if (data['remaining_ticks']<0).any() or (data['tick']>600).any():raise RuntimeError('Trace budget mismatch')
                    if data['actual_arm_collision_known'].any():raise RuntimeError('Uninstrumented physical collision claim')
                for suffix in ['','.telemetry','.trace','.inference']:
                    artifact=path if not suffix else path.with_suffix(suffix+'.json')
                    name=str(artifact.relative_to(ROOT))
                    if name in summary['output_sha'] and sha256(artifact)!=summary['output_sha'][name]:
                        raise RuntimeError('Raw output SHA mismatch '+name)
                for role,digest in d['checkpoint_sha256'].items():
                    if sha256(d['checkpoints'][role])!=digest:raise RuntimeError('Weight mismatch')
                if test not in ['fixed','random','full_loop']:
                    source=RESULTS/f'cohorts/test/seed{seed}/{test}.h5'
                    if sha256(source)!=d['cohort_sha256']:raise RuntimeError('Wrong paired cohort')
                    with h5py.File(source) as h:
                        if len(h)!=100 or set(h)!=set(identities):raise RuntimeError('Formal cohort quota/order')
                        if test=='natural_severe' and not all(g.attrs['natural_source_episode_failed'] for g in h.values()):raise RuntimeError('Natural proxy found')
                means.setdefault((method,test),[]).append(d['success'])
                counts['formal_tests']=counts.get('formal_tests',0)+1
    counts['aggregate_statistics'],counts['paired_holm_tests']=independent_statistics(means,summary)
    if trials!=30500 or summary['total_formal_episode_trials']!=trials:raise RuntimeError('Formal trial total')
    if summary['recorded_physical_interactions']!=interactions:raise RuntimeError('Physical interaction total')
    counts['formal_episode_trials']=trials
    identities={};plan_sha=sha256(RESULTS/'collection_plan.json')
    for split,quota in [('validation',32),('test',100)]:
        ids=set()
        for seed in range(5):
            directory=RESULTS/f'cohorts/{split}/seed{seed}';manifest=read(directory/'manifest.json')
            for case in ['natural_severe']+art+GRADES:
                row=manifest[case];path=directory/(case+'.h5');source=row['source']
                if sha256(path)!=row['sha256'] or sha256(source)!=row['source_sha256']:
                    raise RuntimeError('Cohort/source evidence SHA mismatch')
                with h5py.File(path) as h,h5py.File(source) as original:
                    if len(h)!=quota or set(h)!=set(original):raise RuntimeError('Independent cohort quota')
                    if h.attrs['collection_plan_sha256']!=plan_sha:raise RuntimeError('Collection plan SHA mismatch')
                    for key in h:
                        if set(h[key])!=set(original[key]):raise RuntimeError('Cohort payload field mismatch')
                        for field in h[key]:
                            if not np.array_equal(h[key][field][:],original[key][field][:]):raise RuntimeError('Cohort field changed')
                        if case=='natural_severe':
                            g=h[key]
                            if not bool(g.attrs['natural_source_episode_failed']):raise RuntimeError('Unfailed natural source')
                            identity=tuple(int(g.attrs[k]) for k in ['reference_model_seed','reset_seed','clone','source_episode'])
                            if identity in ids:raise RuntimeError('Repeated natural source episode')
                            ids.add(identity)
                counts['independent_cohort_files']=counts.get('independent_cohort_files',0)+1
        identities[split]=ids
    if identities['validation']&identities['test']:raise RuntimeError('Validation/test leakage')
    if len(identities['validation'])!=160 or len(identities['test'])!=500:raise RuntimeError('Natural unique source quota')
    selection=read(RESULTS/'selection.json');scores={}
    for retreat in ['R0','R1','R2','R3']:
        values=[]
        for seed in range(5):
            rows=read(RESULTS/f'validation/B_{retreat}_seed{seed}/natural_severe.telemetry.json')
            values.append([sum(any(p.get('safe_retreat') for p in row['planner']) for row in rows)/len(rows),
                sum(any(p.get('reached_pregrasp') for p in row['planner']) for row in rows)/len(rows),
                sum(row['success'] for row in rows)/len(rows)])
        scores[retreat]=np.mean(values,axis=0).tolist()
        for i,value in enumerate(scores[retreat]):equal_number(selection['validation_scores'][retreat][i],value,'Selection validation score')
    if selection['selected_by']!='validation_only' or selection['retreat']!=max(scores,key=lambda r:tuple(scores[r])):
        raise RuntimeError('Validation-only selection mismatch')
    for filename,field in [('code_manifest.json','code_manifest_sha'),('protocol.json','protocol_sha'),('selection.json','selection_sha')]:
        if sha256(RESULTS/filename)!=summary[field]:raise RuntimeError('Summary provenance mismatch '+filename)
    for p in (RESULTS/'jobs').rglob('*.failed.json'):raise RuntimeError('Unresolved failed job '+str(p))
    for name in ['per_seed','aggregate','stage','router','failure','distance_bins','paired','timing','deployment']:
        if not (RESULTS/f'metrics/{name}.csv').exists():raise RuntimeError('Missing CSV '+name)
    for row in summary['comparisons']:
        if row['exact_signflip_p']<.0625:raise RuntimeError('Impossible exact five-seed p')
    for filename,keys,reference,fields in [('aggregate.csv',('method','test'),summary['aggregates'],('mean','low','high','sd')),
            ('paired.csv',('left','right','test'),summary['comparisons'],('mean','low','high','sd','exact_signflip_p','holm_exact_p'))]:
        with (RESULTS/'metrics'/filename).open(newline='') as f:rows=list(csv.DictReader(f))
        index={tuple(r[k] for k in keys):r for r in rows}
        if len(index)!=len(reference):raise RuntimeError('CSV endpoint coverage '+filename)
        for expected in reference:
            key=tuple(expected[k] for k in keys)
            for field in fields:equal_number(index[key][field],expected[field],filename+' '+field)
    for seed in range(5):
        for role,row in read(RESULTS/f'deployment_latency/seed{seed}.json').items():
            if row['batch']!=1 or row['concurrent_robot_workers_on_gpu']!=1 or row['simulation_running_on_gpu']:
                raise RuntimeError('Deployment timing scope mismatch')
            values=[s['rpc_s'] for s in row['samples']]
            if len(values)!=100:raise RuntimeError('Deployment sample quota')
            for p in [50,95,99]:equal_number(row[f'p{p}_rpc_s'],np.quantile(values,p/100),'Exclusive latency percentile')
            if row['budget_met']!=all(v<=row['replan_budget_s'] for v in values):raise RuntimeError('Deployment budget mismatch')
            counts['exclusive_latency_roles']=counts.get('exclusive_latency_roles',0)+1
    save_json(RESULTS/'final_audit.json',dict(passed=True,checks=counts,rl_updates=0,
        reports={name:sha256(ROOT/'docs'/name) for name in ['phase15_2_failure_stage_analysis.md','phase15_2_recovery_chain_report.md']}))

if __name__=='__main__':main()
