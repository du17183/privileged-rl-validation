"""Seed-level inference, fixed-condition comparisons and quality heterogeneity."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, ARMS, prepared


def summarize(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else 0.
    margin=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else 0.
    return dict(n=n,mean=mean,sd=sd,variance=sd**2,ci95=[mean-margin,mean+margin],values=x.tolist())


def paired(left,right):
    difference=np.asarray(left)-np.asarray(right)
    signs=np.asarray(list(itertools.product((-1,1),repeat=len(difference))))
    actual=abs(difference.mean());null=np.abs((signs*difference).mean(1))
    exact=float(np.mean(null>=actual-1e-12))
    if np.all(difference==0): p=1.
    else: p=float(stats.ttest_1samp(difference,0).pvalue)
    return dict(**summarize(difference),exact_signflip_p=exact,paired_t_p=p)


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);adjusted=[None]*len(values);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,values[index]*(len(values)-rank)));adjusted[index]=running
    return adjusted


def csv_rows(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def main():
    if not (OUT/'heldout_completed.json').exists(): raise RuntimeError('Independent tests incomplete')
    endpoint_data={};rows=[];curves={};quality={};exposures={}
    for arm in ARMS:
        for seed in range(5):
            key=f'{arm}_seed{seed}';run=f'P11{arm}_seed{seed}'
            held=json.loads((OUT/'heldout'/f'{key}.json').read_text())
            lookup={(r['endpoint'],r['metrics']['condition'],r['metrics']['mode']):r for r in held['results']}
            endpoint_data[key]=lookup
            marker=json.loads((CKPT/run/'completed.json').read_text())
            curve=csv_rows(OUT/f'eval_{run}_level2.csv');nominal=csv_rows(OUT/f'eval_{run}_nominal.csv')
            x=np.array([float(r['env_steps']) for r in curve]);y=np.array([float(r['success']) for r in curve])
            assert x[0]==0 and x[-1]==300000 and len(x)==31
            auc=float(np.trapz(y,x)/300000)
            final=lookup[('final','level2','policy')]['metrics'];best=lookup[('best','level2','policy')]['metrics']
            episodes=csv_rows(OUT/f'episodes_{run}.csv')
            q=np.array([float(r['quality_score']) for r in episodes]);succ=np.array([float(r['success']) for r in episodes])
            bins={'full_score':float(np.mean(q>=.999999)), 'low_lt_0p25':float(np.mean(q<.25)),
                  'partial_0p25_0p75':float(np.mean((q>=.25)&(q<.75))), 'near_0p75_1':float(np.mean((q>=.75)&(q<.999999)))}
            by_level={str(level):dict(episodes=sum(int(r['curriculum_level'])==level for r in episodes),
                transitions=sum(int(r['episode_steps']) for r in episodes if int(r['curriculum_level'])==level),
                successes=sum(float(r['success']) for r in episodes if int(r['curriculum_level'])==level)) for level in range(4)}
            quality[key]=dict(mean=float(q.mean()),sd=float(q.std()),histogram=np.histogram(q,bins=np.linspace(0,1,11))[0].tolist(),
                fractions=bins,quality_is_terminal_outcome_score=True,completed_episode_success=float(succ.mean()))
            exposures[key]=by_level
            late=[r for r in episodes if int(r['env_steps'])>200000]
            row=dict(arm=arm,seed=seed,auc=auc,final_success=final['success'],best_success=best['success'],gap=best['success']-final['success'],
                nominal_final_success=lookup[('final','nominal','policy')]['metrics']['success'],
                nominal_best_success=lookup[('best','nominal','policy')]['metrics']['success'],
                online_successes=marker['online_successes'],online_episodes=marker['online_episodes'],
                online_success_ratio=marker['online_successes']/marker['online_episodes'],
                late_online_successes=int(sum(float(r['success']) for r in late)),
                rollback_events=marker['rejections'],accepted_blocks=marker['acceptances'],best_step=marker['best_step'],
                final_curriculum_level=marker['final_level'],eval_interactions=marker['cumulative_eval_steps'],
                training_interactions=marker['steps'],effective_std=final['effective_std'],anchor_kl=final['anchor_kl'],
                completed_quality_sd=quality[key]['sd'],full_quality_fraction=bins['full_score'])
            rows.append(row);curves[key]=dict(random=curve,nominal=nominal)
    summaries={}
    metrics=[k for k in rows[0] if k not in ('arm','seed')]
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        summaries[arm]={k:summarize([r[k] for r in selected]) for k in metrics}
    contrasts=[]
    for left,right in (('B','A'),('C','B'),('D','C'),('D','A')):
        result=dict(contrast=f'{left}-{right}',metrics={})
        for metric in ('auc','final_success','best_success','nominal_final_success','gap'):
            result['metrics'][metric]=paired([r[metric] for r in rows if r['arm']==left],[r[metric] for r in rows if r['arm']==right])
        contrasts.append(result)
    for metric in ('auc','final_success'):
        adjusted=holm([r['metrics'][metric]['paired_t_p'] for r in contrasts])
        exact=holm([r['metrics'][metric]['exact_signflip_p'] for r in contrasts])
        for i,r in enumerate(contrasts):r['metrics'][metric].update(holm_paired_t_p=adjusted[i],holm_exact_p=exact[i])
    conditions=sorted({c for e,c,m in next(iter(endpoint_data.values()))})
    generalization={}
    for arm in ARMS:
        generalization[arm]={endpoint:{condition:{mode:summarize([endpoint_data[f'{arm}_seed{s}'][(endpoint,condition,mode)]['metrics']['success'] for s in range(5)])
            for mode in ('policy','deterministic') if (endpoint,condition,mode) in endpoint_data[f'{arm}_seed0']}
            for condition in conditions} for endpoint in ('best','final')}
    frozen={};frozen_rows=[]
    for seed in range(5):
        obj=json.loads((OUT/'heldout'/f'A_seed{seed}_frozen.json').read_text())
        lookup={(r['metrics']['condition'],r['metrics']['mode']):r['metrics']['success'] for r in obj['results']}
        frozen_rows.append(lookup)
    for condition in conditions:
        frozen[condition]={mode:summarize([r[(condition,mode)] for r in frozen_rows]) for mode in ('policy','deterministic') if (condition,mode) in frozen_rows[0]}
    against_frozen={arm:{condition:paired(generalization[arm]['final'][condition]['policy']['values'],frozen[condition]['policy']['values'])
        for condition in conditions} for arm in ARMS}
    slices={}
    for arm in ARMS:
        slices[arm]={}
        for endpoint in ('best','final'):
            data=[endpoint_data[f'{arm}_seed{s}'][(endpoint,'level2','policy')]['metrics']['slices'] for s in range(5)]
            slices[arm][endpoint]={k:dict(per_seed=[d[k] for d in data],
                seed_level=summarize([d[k]['success'] for d in data if d[k]['success'] is not None]),
                total_episodes=sum(d[k]['episodes'] for d in data)) for k in data[0]}
    d_a=next(r for r in contrasts if r['contrast']=='D-A')['metrics']['final_success']
    conditional_pass=(d_a['ci95'][0]>0 and summaries['D']['nominal_final_success']['mean']>=.9 and sum(v>0 for v in d_a['values'])>=4)
    conditional_gate=dict(passed=conditional_pass,rule='D-A randomized final paired t95 lower >0, at least4/5 favorable, D nominal mean >=90%',
        evidence=dict(random_effect=d_a,nominal=summaries['D']['nominal_final_success']),
        status='Run sensor-noise and input-mask diagnosis' if conditional_pass else 'Deferred: D did not meet prespecified randomized improvement and nominal preservation gate')
    (OUT/'conditional_gate.json').write_text(json.dumps(conditional_gate,indent=2))
    training=json.loads((OUT/'training_completed.json').read_text());heldout=json.loads((OUT/'heldout_completed.json').read_text())
    summary=dict(protocol=prepared(),groups=summaries,paired_tests=contrasts,generalization=generalization,frozen_phase9_c=frozen,
        final_vs_frozen=against_frozen,slices=slices,quality=quality,curriculum_exposure=exposures,
        conditional_gate=conditional_gate,training=training,heldout=heldout,
        statistical_notes=['Seed is inferential unit; t CI unbounded, model-dependent with n=5',
            'Two-sided exact sign-flip minimum p is0.0625 for5 nonzero pairs; episode count does not overcome seed replication',
            'Predeclared adaptive controller is identical, realized difficulty exposure may differ; fixed Level2 test is primary',
            'Frozen Phase9 C was retested on exactly the same new physical conditions; Phase10 values are historical background'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'per_seed.json').write_text(json.dumps(rows,indent=2))
    with (OUT/'per_seed.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'curves.json').write_text(json.dumps(curves))
    print(json.dumps({a:{m:summaries[a][m]['mean'] for m in ('auc','final_success','nominal_final_success','online_successes')} for a in ARMS},indent=2))
if __name__=='__main__':main()
