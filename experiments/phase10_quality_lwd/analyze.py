"""Five paired seeds; genuine online episode counts; physical-score/Q diagnosis."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
import h5py
from scipy.stats import spearmanr
from experiments.phase10_quality_lwd.protocol import ARMS
from experiments.phase10_quality_lwd.data import expert_records
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase10_quality_lwd'


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def summarize(values):
    a=np.asarray(values,dtype=float)
    if len(a)!=5 or not np.isfinite(a).all():raise RuntimeError('Need five finite paired seeds')
    mean,sd=float(a.mean()),float(a.std(ddof=1))
    half=2.7764451051977987*sd/math.sqrt(5)
    return dict(mean=mean,sd=sd,ci95=[mean-half,mean+half],per_seed=a.tolist())


def paired(x,y):
    d=np.asarray(x)-np.asarray(y)
    p=np.mean([abs(np.mean(d*np.array(signs)))>=abs(d.mean())-1e-12 for signs in itertools.product((-1,1),repeat=5)])
    return dict(difference=summarize(d),exact_signflip_p=float(p))


def main():
    if not (OUT/'training_completed.json').exists() or not (OUT/'heldout_completed.json').exists():
        raise RuntimeError('No final inference before all matched experiments complete')
    heldout={}
    for path in (OUT/'heldout').glob('*_seed?.json'):
        if path.name.startswith('q_diagnosis_'):continue
        for row in json.loads(path.read_text()):heldout[(row['arm'],row['seed'],row['checkpoint'],row['condition'],row['mode'])]=row
    records=[]
    quality=[]
    expert_lengths=np.array([r['length'] for r in expert_records(ROOT/'door_dataset/door_expert_1000.h5')])
    for arm in ARMS:
        for seed in range(5):
            run=f'P10{arm}_seed{seed}'
            marker=json.loads((ROOT/'checkpoints/phase10_quality_lwd'/run/'completed.json').read_text())
            updates=read(OUT/f'updates_{run}.csv')
            episodes=read(OUT/f'episodes_{run}.csv')
            assert marker['steps']==300000 and marker['optimizer_steps']==37500 and len(updates)==30,run
            assert len(episodes)==marker['online_episodes'],run
            assert sum(int(float(e['success'])) for e in episodes)==marker['online_successes'],run
            r=dict(arm=arm,seed=seed,steps=300000,online_successes=marker['online_successes'],
                online_episodes=marker['online_episodes'],online_success_ratio=marker['online_successes']/marker['online_episodes'],
                rollback_events=marker['rollback_events'],rejection_fraction=marker['rejected_blocks']/30,
                evaluation_steps=marker['cumulative_eval_steps'],best_step=marker['best_step'])
            for mode in ('policy','deterministic'):
                rows=read(OUT/f'eval_{run}_{mode}.csv')
                assert len(rows)==31 and int(rows[-1]['env_steps'])==300000,run
                x=np.array([int(e['env_steps']) for e in rows]);y=np.array([float(e['success']) for e in rows])
                r[mode+'_auc']=float(np.trapz(y,x)/300000)
                r[mode+'_curve_final']=float(y[-1])
                r[mode+'_curve_peak']=float(y.max())
                r[mode+'_raw_auc']=float(np.trapz([float(e['raw_candidate_success']) for e in rows],x)/300000)
                best=heldout[(arm,seed,'best','nominal',mode)]
                final=heldout[(arm,seed,'final','nominal',mode)]
                initial=heldout[('REFC',seed,'initial','nominal',mode)]
                r[mode+'_best']=best['success'];r[mode+'_final']=final['success']
                r[mode+'_gap']=best['success']-final['success']
                r[mode+'_frozen_initial']=initial['success']
                r[mode+'_gain_vs_frozen']=final['success']-initial['success']
                for name in ('return','episode_steps','max_angle','final_angle','progress','regression_event'):
                    r[mode+'_final_'+name]=final[name]
                    r[mode+'_delta_'+name]=final[name]-initial[name]
                r[mode+'_std']=final['effective_std'];r[mode+'_kl']=final['anchor_kl']
            distribution=marker['replay_distribution'];draws=distribution['draws'];total=sum(draws.values())
            assert total==37500*256,run
            for source,count in draws.items():r['sample_'+source+'_fraction']=count/total
            r['quality_density_max']=distribution['transition_density_ratio_max']
            r['selected_fraction']=distribution['selected_fraction']
            with h5py.File(ROOT/'datasets/phase10_quality_lwd'/f'{run}.h5','r') as h5:
                lengths=np.r_[expert_lengths,h5['trajectories/length'][:]]
                scores=np.r_[np.ones(1000),h5['trajectories/quality_score'][:]].astype(np.float32)
                uniform=lengths/lengths.sum()
                if ARMS[arm]['mode']=='weighted':
                    targeted=lengths*np.exp((scores-1)/ARMS[arm]['temperature'])
                    probability=.1*uniform+.9*targeted/targeted.sum()
                elif ARMS[arm]['mode']=='selected':
                    selected=scores>=np.quantile(scores,.8)
                    targeted=lengths*selected
                    probability=.2*uniform+.8*targeted/targeted.sum()
                else:probability=uniform
                r['final_quality_mass_total_variation']=float(np.abs(probability-uniform).sum()/2)
                r['completed_buffer_quality_full_fraction']=float(np.mean(scores>=1-1e-6))
            windows=[sum(float(e['success']) for e in episodes if start<int(e['env_steps'])<=start+10000)
                     for start in range(0,300000,10000)]
            r['online_success_windows']=sum(v>0 for v in windows)
            records.append(r)
            score=np.array([float(e['quality_score']) for e in episodes])
            mc=np.array([float(e['discounted_return']) for e in episodes])
            q=np.array([float(e['initial_q']) for e in episodes])
            def rho(a,b):
                return float(spearmanr(a,b).statistic) if np.ptp(a)>1e-8 and np.ptp(b)>1e-8 else None
            quality.append(dict(arm=arm,seed=seed,episodes=len(episodes),quality_mean=float(score.mean()),
                quality_sd=float(score.std()),quality_quantiles=np.quantile(score,[0,.1,.5,.9,1]).tolist(),
                full_quality_fraction=float(np.mean(score>=1-1e-6)),
                quality_vs_discounted_return_spearman=rho(score,mc),initial_q_vs_discounted_return_spearman=rho(q,mc),
                caveat='Quality includes completed success/progress; Q initial online estimates change during learning; this descriptive MC association is not an early prediction or policy-improvement proof.'))
    summaries={arm:{key:summarize([r[key] for r in records if r['arm']==arm])
                   for key in records[0] if key not in ('arm','seed')} for arm in ARMS}
    comparisons={}
    for arm,control in [('A','REFC'),('B','REFC'),('D05','B'),('D10','B'),('D20','B'),('E','B')]:
        comparisons[f'{arm}-{control}']={key:paired(summaries[arm][key]['per_seed'],summaries[control][key]['per_seed'])
            for key in ('policy_auc','policy_final','policy_gap','online_success_ratio','rollback_events','policy_final_return','policy_final_episode_steps')}
    # Secondary recipe comparisons change both source quotas/eligibility and
    # quality rules. They describe the deployed continuation recipe, rather
    # than isolating quality; the preregistered D/E versus B tests remain primary.
    for arm in ('D05','D10','D20','E'):
        comparisons[f'{arm}-REFC']={key:paired(summaries[arm][key]['per_seed'],summaries['REFC'][key]['per_seed'])
            for key in ('policy_auc','policy_final','policy_gap','online_success_ratio','rollback_events','policy_final_return','policy_final_episode_steps')}
    for arm in ARMS:
        comparisons[f'{arm}-frozen']={key:paired(summaries[arm]['policy_'+key]['per_seed'],summaries[arm]['policy_frozen_initial']['per_seed'])
                                    for key in ('final',)}
    # Holm multiplicity adjustment separately for three temperature AUC/final tests.
    for metric in ('policy_auc','policy_final'):
        ranked=sorted((comparisons[f'{a}-B'][metric]['exact_signflip_p'],a) for a in ('D05','D10','D20'))
        previous=0.
        for i,(p,a) in enumerate(ranked):
            previous=max(previous,min(1.,(3-i)*p))
            comparisons[f'{a}-B'][metric]['holm_p_temperature_family']=previous
    grouped={}
    for (arm,seed,checkpoint,condition,mode),row in heldout.items():
        grouped.setdefault((arm,checkpoint,condition,mode),{})[seed]=row
    tests=[]
    for (arm,checkpoint,condition,mode),rows in sorted(grouped.items()):
        assert set(rows)==set(range(5))
        tests.append(dict(arm=arm,checkpoint=checkpoint,condition=condition,mode=mode,
                          success=summarize([rows[s]['success'] for s in range(5)]),
                          initial_robot_obs_delta=summarize([rows[s]['initial_robot_obs_max_abs_delta'] for s in range(5)]),
                          initial_gt_delta=summarize([rows[s]['initial_gt_max_abs_delta'] for s in range(5)])))
    gates={}
    for arm in ('D10','E'):
        s=summaries[arm]
        perturb=[t['success']['mean'] for t in tests if t['arm']==arm and t['checkpoint']=='final' and t['condition']!='nominal' and t['mode']=='policy']
        c=comparisons[f'{arm}-B']
        checks=dict(quality_auc_gain_ci_positive=c['policy_auc']['difference']['ci95'][0]>0,
            final_no_material_drop=c['policy_final']['difference']['ci95'][0]>=-.05,
            continual_online_success_each_seed=all(v>=3 for v in s['online_success_windows']['per_seed']),
            final_mean_ge_90=s['policy_final']['mean']>=.9,gap_mean_le_05=s['policy_gap']['mean']<=.05,
            rejections_each_le_20pct=all(v<=.2 for v in s['rejection_fraction']['per_seed']),
            perturbations_mean_ge_80=bool(perturb) and min(perturb)>=.8)
        gates[arm]=dict(criteria=checks,ready_for_full_lwd_divl=all(checks.values()),
            statistical_limit='Five pairs exact two-sided sign-flip min p=.0625; positive CI is exploratory, not a p<.05 claim')
    for name,value in [('summary',summaries),('paired_tests',comparisons),('quality_diagnosis',quality),('heldout_summary',tests),('gates',gates),('per_seed',records)]:
        (OUT/f'{name}.json').write_text(json.dumps(value,indent=2))
    with (OUT/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    frozen_quality=[]
    for arm in ARMS:
        for seed in range(5):
            rows=json.loads((OUT/'heldout'/f'q_diagnosis_{arm}_seed{seed}.json').read_text())
            assert len(rows)==64
            def rho(x,y):
                a=np.array([r[x] for r in rows]);b=np.array([r[y] for r in rows])
                return float(spearmanr(a,b).statistic) if np.ptp(a)>1e-8 and np.ptp(b)>1e-8 else None
            frozen_quality.append(dict(arm=arm,seed=seed,episodes=64,
                success_fraction=np.mean([r['success'] for r in rows]),
                quality_full_fraction=np.mean([r['quality_score']>=1-1e-6 for r in rows]),
                quality_vs_reward_mc_rho=rho('quality_score','reward_mc'),
                q_vs_reward_mc_rho=rho('initial_q','reward_mc'),q_vs_soft_mc_rho=rho('initial_q','soft_mc'),
                q_soft_mc_abs_error=np.mean([abs(r['initial_q']-r['soft_mc']) for r in rows]),
                q_soft_mc_bias=np.mean([r['initial_q']-r['soft_mc'] for r in rows]),
                initial_q_mean=np.mean([r['initial_q'] for r in rows]),soft_mc_mean=np.mean([r['soft_mc'] for r in rows]),
                twin_q_disagreement=np.mean([abs(r['initial_q1']-r['initial_q2']) for r in rows]),
                note='Fixed deployed stochastic actor, fixed checkpoint alpha; soft MC proxy includes discounted conditional expected entropy from next action onward, unlike reward-only MC. Q estimates a policy expectation; noisy single-trajectory correlations do not prove calibration or miscalibration. Scores are completed outcomes, not predictive values.'))
    (OUT/'frozen_value_diagnosis.json').write_text(json.dumps(frozen_quality,indent=2))
    print('All five-seed paired analyses complete')

if __name__=='__main__':main()
