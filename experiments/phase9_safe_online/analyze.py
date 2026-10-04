"""Seed-level paired inference, accepted-policy curves and independent endpoints."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
from experiments.phase9_safe_online.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def summarize(values):
    a=np.asarray(values,dtype=float)
    if len(a)!=5:
        raise RuntimeError('Inference requires all five seeds')
    mean,sd=float(a.mean()),float(a.std(ddof=1))
    half=2.7764451051977987*sd/math.sqrt(5)
    return dict(mean=mean,sd=sd,variance=sd*sd,ci95=[mean-half,mean+half],per_seed=a.tolist())

def paired(x,y):
    difference=np.asarray(x)-np.asarray(y)
    p=np.mean([abs(np.mean(difference*np.array(signs)))>=abs(difference.mean())-1e-12
               for signs in itertools.product((-1,1),repeat=5)])
    return dict(difference=summarize(difference),exact_signflip_p_two_sided=float(p))

def main():
    if not (OUT/'training_completed.json').exists() or not (OUT/'heldout_completed.json').exists():
        raise RuntimeError('Do not finalize incomplete experiments')
    independent={}
    for path in (OUT/'heldout').glob('*_seed?.json'):
        for row in json.loads(path.read_text()):
            independent[(row['arm'],row['seed'],row['checkpoint'],row['condition'],row['mode'])]=row
    records=[]
    for arm in ARMS:
        for seed in range(5):
            run=f'P9{arm}_seed{seed}'
            marker=json.loads((ROOT/'checkpoints/phase9_safe_online'/run/'completed.json').read_text())
            updates=read(OUT/f'updates_{run}.csv')
            episodes=read(OUT/f'episodes_{run}.csv')
            assert len(updates)==30 and marker['steps']==300000 and marker['optimizer_steps']==37500,run
            assert len(episodes)==marker['online_episodes'] and sum(int(float(e['success'])) for e in episodes)==marker['online_successes'],run
            record=dict(arm=arm,seed=seed,online_successes=marker['online_successes'],online_episodes=marker['online_episodes'],
                online_success_ratio=marker['online_successes']/max(1,marker['online_episodes']),
                first_online_success_step=marker['first_online_success_step'],
                rejections=marker['rejected_blocks'],accepted_blocks=marker['accepted_blocks'],
                rejection_fraction=marker['rejected_blocks']/30,evaluation_steps=marker['cumulative_eval_steps'],
                source_anchor_step=marker['source_anchor_step'],best_step=marker['best_step'])
            for mode,prefix in [('deterministic','det'),('policy','policy')]:
                rows=read(OUT/f'eval_{run}_{mode}.csv')
                assert len(rows)==31 and int(rows[-1]['env_steps'])==300000,run
                steps=np.array([int(r['env_steps']) for r in rows])
                success=np.array([float(r['success']) for r in rows])
                record[prefix+'_auc']=float(np.trapz(success,steps)/300000)
                raw=np.array([float(r['raw_candidate_success']) for r in rows])
                record[prefix+'_raw_candidate_auc']=float(np.trapz(raw,steps)/300000)
                record[prefix+'_raw_candidate_final']=float(raw[-1])
                record[prefix+'_curve_best']=float(success.max())
                record[prefix+'_curve_final']=float(success[-1])
                best=independent[(arm,seed,'best','nominal',mode)]['success']
                final=independent[(arm,seed,'final','nominal',mode)]['success']
                record[prefix+'_best']=best
                record[prefix+'_final']=final
                record[prefix+'_gap']=best-final
                initial_arm='A' if arm in ('A','B') else 'C'
                initial=independent[(initial_arm,seed,'initial','nominal',mode)]['success']
                record[prefix+'_cap_matched_anchor']=initial
                record[prefix+'_gain_over_frozen']=final-initial
                endpoint=independent[(arm,seed,'final','nominal',mode)]
                frozen=independent[(initial_arm,seed,'initial','nominal',mode)]
                for name in ('episode_steps','return','progress','max_angle','final_angle','regression_event'):
                    record[prefix+'_final_'+name]=endpoint[name]
                    record[prefix+'_anchor_'+name]=frozen[name]
                    record[prefix+'_delta_'+name]=endpoint[name]-frozen[name]
                record[prefix+'_final_effective_std']=float(rows[-1]['effective_std'])
                record[prefix+'_final_raw_std']=float(rows[-1]['raw_std'])
                record[prefix+'_final_kl']=float(rows[-1]['anchor_kl'])
            record['noise001_final']=independent[(arm,seed,'final','nominal','noise001')]['success']
            windows=[]
            for start in range(0,300000,10000):
                completed=[e for e in episodes if start<int(e['env_steps'])<=start+10000]
                windows.append(dict(start=start,end=start+10000,episodes=len(completed),successes=sum(int(float(e['success'])) for e in completed)))
            record['successful_windows']=sum(w['successes']>0 for w in windows)
            (OUT/f'online_windows_{run}.json').write_text(json.dumps(windows,indent=2))
            records.append(record)
    summaries={}
    numeric=[k for k in records[0] if k not in ('arm','seed','first_online_success_step')]
    for arm in ARMS:
        seeds=[r for r in records if r['arm']==arm]
        summaries[arm]={key:summarize([r[key] for r in seeds]) for key in numeric}
        summaries[arm]['first_online_success_step']=[r['first_online_success_step'] for r in seeds]
    comparisons={}
    for arm,control in [('B','A'),('C','B'),('C','A'),('D','C'),('CANN','C'),('D100','D'),('D30','D')]:
        comparisons[f'{arm}-{control}']={key:paired(summaries[arm][key]['per_seed'],summaries[control][key]['per_seed'])
            for key in ('policy_auc','det_auc','det_raw_candidate_auc','policy_raw_candidate_auc','policy_final','det_final','policy_gap','online_success_ratio','rejection_fraction')}
    for arm in ARMS:
        comparisons[f'{arm}-cap_matched_frozen_anchor']={mode:paired(summaries[arm][mode+'_final']['per_seed'],summaries[arm][mode+'_cap_matched_anchor']['per_seed']) for mode in ('policy','det')}
        for name in ('episode_steps','return','progress','max_angle','final_angle','regression_event'):
            comparisons[f'{arm}-cap_matched_frozen_anchor']['policy_'+name]=paired(summaries[arm]['policy_final_'+name]['per_seed'],summaries[arm]['policy_anchor_'+name]['per_seed'])
    grouped={}
    for key,row in independent.items():
        arm,seed,selection,condition,mode=key
        grouped.setdefault((arm,selection,condition,mode),{})[seed]=row
    heldout=[]
    for key,seeds in sorted(grouped.items()):
        assert set(seeds)==set(range(5)),key
        arm,selection,condition,mode=key
        heldout.append(dict(arm=arm,checkpoint=selection,condition=condition,mode=mode,
            success=summarize([seeds[s]['success'] for s in range(5)]),
            progress=summarize([seeds[s]['progress'] for s in range(5)]),
            episode_steps=summarize([seeds[s]['episode_steps'] for s in range(5)]),
            return_value=summarize([seeds[s]['return'] for s in range(5)]),
            regression_rate=summarize([seeds[s]['regression_event'] for s in range(5)])))
    gates={}
    for arm in ('A','B','C','D'):
        s=summaries[arm]
        perturb=[r['success']['mean'] for r in heldout if r['arm']==arm and r['checkpoint']=='final' and r['condition']!='nominal']
        criteria=dict(online_success_all_seeds=all(v>0 for v in s['online_successes']['per_seed']),
            online_success_in_multiple_windows=all(v>=3 for v in s['successful_windows']['per_seed']),
            final_stochastic_mean_ge_90=s['policy_final']['mean']>=.9,
            best_final_mean_gap_le_05=s['policy_gap']['mean']<=.05,
            best_final_each_gap_le_10=all(v<=.1 for v in s['policy_gap']['per_seed']),
            rejections_each_le_20pct=all(v<=.2 for v in s['rejection_fraction']['per_seed']),
            light_noise_mean_ge_90=s['noise001_final']['mean']>=.9,
            perturbation_each_condition_mean_ge_80=bool(perturb) and min(perturb)>=.8)
        gains=s['policy_gain_over_frozen']['ci95']
        gates[arm]=dict(criteria=criteria,engineering_gate=all(criteria.values()),
            demonstrated_improvement_over_cap_matched_anchor=gains[0]>0,
            lwd_divl_ready=all(criteria.values()),
            readiness_basis='User engineering stability/generalization criteria; incremental success gain is separate research evidence, not an extra success-ceiling gate')
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    (OUT/'paired_tests.json').write_text(json.dumps(comparisons,indent=2))
    (OUT/'heldout_summary.json').write_text(json.dumps(heldout,indent=2))
    # The unchanged task has only success and 600-step timeout terminations.
    # Verify this failure-duration assumption against every training episode.
    failure_lengths={int(e['episode_steps']) for arm in ARMS for seed in range(5)
        for e in read(OUT/f'episodes_P9{arm}_seed{seed}.csv') if not float(e['success'])}
    if failure_lengths != {600}:
        raise RuntimeError('Conditional completion-time reconstruction assumption failed')
    speed={}
    for arm in ARMS:
        initial_arm='A' if arm in ('A','B') else 'C'
        initial_times=[];final_times=[]
        for seed in range(5):
            original=independent[(initial_arm,seed,'initial','nominal','policy')]
            final=independent[(arm,seed,'final','nominal','policy')]
            def duration(r):
                return (r['episode_steps']-600*(1-r['success']))/r['success'] if r['success']>0 else None
            initial_times.append(duration(original));final_times.append(duration(final))
        if all(v is not None for v in initial_times+final_times):
            speed[arm]=dict(anchor=summarize(initial_times),final=summarize(final_times),
                paired_final_minus_anchor=paired(final_times,initial_times),
                interpretation='Exploratory conditional successful-episode duration; derived from mean duration and success fraction under verified 600-step failed episodes. Not an extra readiness gate.')
        else:
            speed[arm]=dict(anchor_per_seed=initial_times,final_per_seed=final_times,not_estimable_for_all_five_seeds=True)
    (OUT/'successful_completion_time.json').write_text(json.dumps(speed,indent=2))
    (OUT/'readiness.json').write_text(json.dumps(gates,indent=2))
    with (OUT/'per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    cost=dict(training_interactions=10500000,
        formal_and_guard_interactions=sum(r['evaluation_steps'] for r in records),
        independent_interactions=sum(r['eval_env_steps'] for r in independent.values()),
        independent_episodes=sum(r['episodes'] for r in independent.values()),
        preparation_interactions_reconstructed_from_logs_and_datasets=True,
        preparation_note='Initial preparation generated all 64 datasets but failed summary serialization; recovery reused data. Recovered deterministic eval step count was reconstructed from ordering and is not exact. Do not merge that estimate into exact training/evaluation counts.',
        pilot_interactions=30240,
        pilot_evaluation_interactions=sum(json.loads(p.read_text())['cumulative_eval_steps'] for p in (ROOT/'checkpoints/phase9_safe_online').glob('*_smoke/completed.json')))
    cost['perturbation_pilot_evaluation_interactions']=sum(r['eval_env_steps'] for p in (OUT/'heldout_pilot').glob('*_seed?.json') for r in json.loads(p.read_text()))
    (OUT/'interaction_cost.json').write_text(json.dumps(cost,indent=2))
    print({a:{k:round(s[k]['mean'],4) for k in ('policy_auc','policy_final','policy_gap','online_successes','rejection_fraction','policy_gain_over_frozen')} for a,s in summaries.items()})

if __name__=='__main__':
    main()
