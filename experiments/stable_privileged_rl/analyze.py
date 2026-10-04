"""Five-seed Phase 4 analysis; strict completeness checks and paired tests."""

import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'stable_privileged_rl'
VARIANTS = ('E0','E25','E50','E100','E200','E100RB','E100R1','E100M','Q0','QGT','QV')
COMPARISONS = (('E0','E25'),('E0','E50'),('E0','E100'),('E0','E200'),
               ('E100','E100RB'),('E100','E100R1'),('E100RB','E100R1'),
               ('E100','E100M'),('Q0','QGT'),('QGT','QV'))
METRICS = ('auc','auc_all_interactions','raw_auc','final_success','raw_final_success',
           'best_success','best_minus_final','degradation_ratio','late_success','max_drawdown')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, seed=20260929):
    a = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).choice(a, size=(20000,len(a)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025,0.975])]


def signflip(diff):
    a = np.asarray(diff, dtype=float)
    observed = abs(float(a.mean()))
    return sum(abs(float(np.mean(a*np.asarray(signs)))) >= observed-1e-12
               for signs in itertools.product((-1,1), repeat=len(a))) / 2**len(a)


def training():
    curves = {}
    rows = []
    thresholds = []
    for variant in VARIANTS:
        for seed in range(5):
            source = OUT/f'eval_{variant}_seed{seed}.csv'
            data = read_csv(source)
            if len(data) != 51:
                raise ValueError(f'Expected 51 validation points, found {len(data)}: {source}')
            x = np.array([int(r['env_steps']) for r in data])
            y = np.array([float(r['success_rate']) for r in data])
            reward = np.array([float(r['mean_return']) for r in data])
            raw = np.array([float(r['pre_rollback_success']) for r in data])
            total_x=x+np.array([int(r['eval_env_steps']) for r in data])
            if x[0] != 0 or x[-1] != 500000 or np.any(np.diff(x) <= 0):
                raise ValueError(f'Incomplete/nonmonotonic curve: {source}')
            curves[(variant,seed)] = (x,y,reward)
            for target in (.5,.8,.9):
                hits = np.flatnonzero(y >= target)
                stable = next((int(x[i]) for i in range(len(y))
                               if np.all(y[i:] >= target)), None)
                thresholds.append({'variant':variant,'seed':seed,'target_success':target,
                                   'first_steps':int(x[hits[0]]) if len(hits) else '',
                                   'first_total_interactions':int(total_x[hits[0]]) if len(hits) else '',
                                   'sustained_steps':stable if stable is not None else '',
                                   'sustained_total_interactions':int(total_x[np.flatnonzero(x==stable)[0]]) if stable is not None else '',
                                   'ever_reached':int(bool(len(hits))),
                                   'sustained_to_500k':int(stable is not None)})
            best = float(y.max())
            drop = best-float(y[-1])
            first_high = np.flatnonzero(y>=.8)
            first_raw_high = np.flatnonzero(raw>=.8)
            rows.append({'variant':variant,'seed':seed,
                         'auc':float(np.trapz(y,x)/500000),
                         'raw_auc':float(np.trapz(raw,x)/500000),
                         'auc_all_interactions':float(np.trapz(np.r_[y[0],y],
                             np.r_[0,total_x])/total_x[-1]),
                         'final_success':float(y[-1]),
                         'raw_final_success':float(raw[-1]),
                         'best_success':best,
                         'best_steps':int(x[int(np.argmax(y))]),
                         'best_minus_final':drop,
                         'degradation_ratio':drop/max(best,1e-8),
                         'max_drawdown':float(np.max(np.maximum.accumulate(y)-y)),
                         'late_success':float(y[x>=400000].mean()),
                         'reward_auc':float(np.trapz(reward,x)/500000),
                         'evaluation_interactions':int(data[-1]['eval_env_steps']),
                         'total_interactions_including_evaluation':500000+int(data[-1]['eval_env_steps']),
                         'rollback_count':int(data[-1]['rollback_count']),
                         'reached_80':int(bool(np.any(y>=.8))),
                         'catastrophe_after_80_deployed':int(bool(len(first_high) and
                             np.any(y[first_high[0]+1:]<.2))),
                         'catastrophe_after_80_raw':int(bool(len(first_raw_high) and
                             np.any(raw[first_raw_high[0]+1:]<.2))),
                         'weighted_eval_fraction':float(np.mean([int(r['weighted_replay_active']) for r in data])),
                         'value_rank_rho_mean':float(np.mean([float(r['value_rank_rho']) for r in data
                                                             if r['value_rank_rho'] not in ('','nan')]))
                                               if any(r['value_rank_rho'] not in ('','nan') for r in data) else float('nan')})
    write_csv(OUT/'seed_metrics.csv', rows)
    write_csv(OUT/'thresholds.csv', thresholds)
    aggregate = []
    for variant in VARIANTS:
        subset = [r for r in rows if r['variant']==variant]
        item = {'variant':variant,'seeds':5}
        for metric in METRICS+('reward_auc','evaluation_interactions',
                               'total_interactions_including_evaluation',
                               'rollback_count','weighted_eval_fraction',
                               'reached_80','catastrophe_after_80_deployed',
                               'catastrophe_after_80_raw'):
            a = [r[metric] for r in subset]
            lo,hi = bootstrap(a)
            item[metric+'_mean'] = float(np.mean(a))
            item[metric+'_ci95_low'] = lo
            item[metric+'_ci95_high'] = hi
        aggregate.append(item)
    write_csv(OUT/'aggregate_metrics.csv', aggregate)
    paired = []
    for baseline,variant in COMPARISONS:
        for metric in METRICS:
            diff = [next(r for r in rows if r['variant']==variant and r['seed']==seed)[metric] -
                    next(r for r in rows if r['variant']==baseline and r['seed']==seed)[metric]
                    for seed in range(5)]
            lo,hi = bootstrap(diff)
            paired.append({'comparison':f'{variant}_minus_{baseline}','metric':metric,
                           'mean_difference':float(np.mean(diff)),
                           'ci95_low':lo,'ci95_high':hi,
                           'exact_signflip_p':signflip(diff),
                           'seed_differences':';'.join(f'{d:.6f}' for d in diff)})
    write_csv(OUT/'paired_effects.csv', paired)
    return curves,rows,aggregate,paired


def heldout():
    rows = []
    for variant in VARIANTS:
        for seed in range(5):
            for mode in ('best','final'):
                data = read_csv(OUT/f'heldout_{variant}_seed{seed}_{mode}.csv')
                if len(data)!=1 or int(data[0]['episodes'])!=64:
                    raise ValueError(f'Bad heldout result: {variant}/{seed}/{mode}')
                rows.append(data[0])
    write_csv(OUT/'heldout_all.csv', rows)
    aggregate = []
    for variant in VARIANTS:
        for mode in ('best','final'):
            subset=[r for r in rows if r['variant']==variant and r['mode']==mode]
            success=[float(r['heldout_success_rate']) for r in subset]
            lo,hi=bootstrap(success)
            aggregate.append({'variant':variant,'mode':mode,'seeds':5,
                              'success_mean':float(np.mean(success)),
                              'ci95_low':lo,'ci95_high':hi,
                              'reward_mean':float(np.mean([float(r['heldout_mean_return']) for r in subset])),
                              'angle_prediction_mae_rad':float(np.mean([
                                  float(r['angle_prediction_mae_rad']) for r in subset])),
                              'contact_prediction_accuracy':float(np.mean([
                                  float(r['contact_prediction_accuracy']) for r in subset]))})
    write_csv(OUT/'heldout_aggregate.csv',aggregate)
    paired=[]
    for baseline,variant in COMPARISONS:
        for mode in ('best','final'):
            diff=[float(next(r for r in rows if r['variant']==variant and r['seed']==str(seed) and r['mode']==mode)['heldout_success_rate'])-
                  float(next(r for r in rows if r['variant']==baseline and r['seed']==str(seed) and r['mode']==mode)['heldout_success_rate'])
                  for seed in range(5)]
            lo,hi=bootstrap(diff)
            paired.append({'comparison':f'{variant}_minus_{baseline}','mode':mode,
                           'mean_difference':float(np.mean(diff)),
                           'ci95_low':lo,'ci95_high':hi,'exact_signflip_p':signflip(diff)})
    write_csv(OUT/'heldout_paired_effects.csv',paired)
    return aggregate,paired


def diagnostics():
    rows=[]
    for variant in VARIANTS:
        for seed in range(5):
            data=read_csv(OUT/f'diagnostics_{variant}_seed{seed}.csv')
            if len(data)!=15625:
                raise ValueError(f'Expected 15625 vector steps: {variant}/{seed}: {len(data)}')
            late=[r for r in data if int(r['env_steps'])>=400000]
            q=np.array([float(r['q_mean']) for r in late])
            qvar=np.array([float(r['q_variance']) for r in late])
            loss=np.array([float(r['critic_loss']) for r in late])
            rows.append({'variant':variant,'seed':seed,'late_q_mean':float(np.nanmean(q)),
                         'late_q_variance_median':float(np.nanmedian(qvar)),
                         'late_q_variance_p95':float(np.nanquantile(qvar,.95)),
                         'late_critic_loss_median':float(np.nanmedian(loss)),
                         'late_critic_loss_p95':float(np.nanquantile(loss,.95))})
    write_csv(OUT/'value_diagnostics.csv',rows)
    return rows


def plots(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups={'windows':('E0','E25','E50','E100','E200'),
            'rollback':('E100','E100RB','E100R1'),
            'multitask':('E100','E100M'),
            'value_replay':('Q0','QGT','QV')}
    for group, variants in groups.items():
        for index,suffix,ylabel in ((1,'success','Success rate'),(2,'reward','Mean episode return')):
            fig,ax=plt.subplots(figsize=(8,4.8))
            for variant in variants:
                x=curves[(variant,0)][0]
                y=np.stack([curves[(variant,seed)][index] for seed in range(5)])
                mean=y.mean(axis=0)
                sem=y.std(axis=0,ddof=1)/np.sqrt(5)
                ax.plot(x,mean,label=variant,lw=2)
                ax.fill_between(x,mean-sem,mean+sem,alpha=.13)
            ax.set(xlabel='Online environment steps',ylabel=ylabel)
            if suffix=='success': ax.set_ylim(0,1)
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT/f'{suffix}_curve_{group}.png',dpi=180)
            plt.close(fig)


def main():
    curves,rows,aggregate,paired=training()
    heldout_aggregate,heldout_paired=heldout()
    value=diagnostics()
    plots(curves)
    summary={'protocol':'door_phase4_stable_privileged_v2_independent_evaluation','variants':VARIANTS,
             'new_online_training_interactions':27500000,
             'training_aggregate':aggregate,'training_paired':paired,
             'heldout_aggregate':heldout_aggregate,'heldout_paired':heldout_paired,
             'value_diagnostics':value}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Analyzed {len(rows)} training runs and {len(VARIANTS)*5*2} heldout replays')


if __name__=='__main__':
    main()
