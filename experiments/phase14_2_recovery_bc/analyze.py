"""Seed-level uncertainty, paired comparisons and explicit Anchor gates."""
import csv,itertools,json,math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc'
ARMS=['A','B','C','D','E','F','D_no_handle'];TESTS=['fixed','random','ee_offset','contact_loss','door_regression','stagnation','natural_severe']

def ci(v):
    v=np.asarray(v,float);mean=float(v.mean());sd=float(v.std(ddof=1));margin=float(stats.t.ppf(.975,len(v)-1)*sd/math.sqrt(len(v)))
    return dict(mean=mean,sd=sd,ci95=[mean-margin,mean+margin],seeds=len(v),values=v.tolist())

def paired(x,y):
    d=np.asarray(x)-np.asarray(y);v=ci(d);t=stats.ttest_rel(x,y)
    combinations=np.asarray(list(itertools.product([-1,1],repeat=len(d))))
    p=float(np.mean(np.abs((combinations*d).mean(1))>=abs(d.mean())-1e-12))
    v.update(paired_t_p=float(t.pvalue) if np.isfinite(t.pvalue) else (1.0 if np.all(d==0) else 0.0),
             exact_signflip_p=p,positive_seeds=int((d>0).sum()))
    return v

def main():
    raw={};summary={};rows=[]
    for arm in ARMS:
        summary[arm]={}
        for test in TESTS:
            values=[]
            for seed in range(5):
                file=R/'evaluation'/f'{arm}_seed{seed}'/(test+'.json')
                d=json.loads(file.read_text());raw[arm,seed,test]=d;values.append(d['success'])
                row=dict(arm=arm,seed=seed,test=test,success=d['success'],episodes=d['episodes'],
                         max_angle=d['max_angle_rad'] if 'max_angle_rad' in d else d.get('mean_max_angle_rad'),
                         final_angle=d['final_angle_rad'] if 'final_angle_rad' in d else d.get('mean_final_angle_rad'))
                for k in ['reapproach','recontact','regrasp','progress_resumed']:row[k]=d.get(k)
                rows.append(row)
            summary[arm][test]=ci(values)
        summary[arm]['artificial_mean']=ci([np.mean([raw[arm,s,t]['success'] for t in TESTS[2:6]]) for s in range(5)])
    comparisons={}
    for name,a,b in [('D-B','D','B'),('D-F','D','F'),('D-C','D','C'),('F-B','F','B')]:
        comparisons[name]={t:paired(summary[a][t]['values'],summary[b][t]['values']) for t in ['random','fixed','artificial_mean','natural_severe']}
    ordered=sorted(comparisons,key=lambda k:comparisons[k]['random']['paired_t_p']);previous=0
    for i,name in enumerate(ordered):
        previous=max(previous,min(1.,comparisons[name]['random']['paired_t_p']*(4-i)));comparisons[name]['random']['holm_p']=previous
    gate=dict(random_over80=summary['D']['random']['mean']>.8,
              all_required_paired_effects=all(comparisons[k]['random']['ci95'][0]>0 and comparisons[k]['random']['positive_seeds']==5 and comparisons[k]['random']['holm_p']<.05 for k in ['D-B','D-F','D-C']),
              artificial_improved=comparisons['D-B']['artificial_mean']['ci95'][0]>0,
              fixed_not_degraded=summary['D']['fixed']['mean']>=summary['B']['fixed']['mean']-.05,
              seed_sd_below10=summary['D']['random']['sd']<.10)
    gate['candidate_qualified']=all(gate.values());gate['independent_retest_required']=True;gate['rl_allowed_now']=False
    gate['lwd_divl_ready']=False
    result=dict(summary=summary,paired=comparisons,anchor_gate=gate,
                uncertainty='Student-t95CI across5training seeds; paired identical resets. Exact signflip n5 minimum two-sided p=0.0625. Not episode-as-independent-seed.',
                multiple_comparisons='Holm correction of the four prespecified main random-success t tests',rl_updates=0)
    final={}
    for arm in ARMS[:6]:
        final[arm]=ci([json.loads((R/'evaluation_final'/f'{arm}_seed{s}'/'random.json').read_text())['success'] for s in range(5)])
    result['final_update_random']=final
    result['final_update_paired']={k:paired(final[a]['values'],final[b]['values']) for k,a,b in [('D-B','D','B'),('D-F','D','F'),('D-C','D','C'),('F-B','F','B')]}
    ordered=sorted(result['final_update_paired'],key=lambda k:result['final_update_paired'][k]['paired_t_p']);previous=0
    for i,k in enumerate(ordered):
        previous=max(previous,min(1.,result['final_update_paired'][k]['paired_t_p']*(4-i)));result['final_update_paired'][k]['holm_p']=previous
    gate['final_update_effect_direction']=all(result['final_update_paired'][k]['mean']>0 for k in ['D-B','D-F','D-C'])
    gate['candidate_qualified']=gate['candidate_qualified'] and gate['final_update_effect_direction']
    matched={}
    for test in TESTS[2:]:
        robots=[];states=[]
        for arm in ARMS:
            for seed in range(5):
                with np.load(R/'evaluation'/f'{arm}_seed{seed}'/(test+'.initial.npz')) as h:
                    robots.append(h['robot']);states.append(h['state'])
        rob=np.stack(robots);gt=np.stack(states)
        position_spread=np.linalg.norm(np.ptp(rob[:,:,18:21],axis=0),axis=-1)
        handle_spread=np.linalg.norm(np.ptp(gt[:,:,2:5],axis=0),axis=-1)
        contacts=gt[:,:,9:11]>.5
        matched[test]=dict(max_eef_span_m=float(position_spread.max()),median_eef_span_m=float(np.median(position_spread)),
           max_handle_span_m=float(handle_spread.max()),max_q_span_rad=float(np.ptp(rob[:,:,:9],axis=0).max()),
           contact_consistent_fraction=float(np.mean((contacts==contacts[:1]).all((0,2)))),
           mean_snapshot_distance_m=float(np.linalg.norm(rob[0,:,18:21]-gt[0,:,2:5],axis=-1).mean()))
    result['physical_state_matching']=matched
    quantity={}
    for arm in ['E_S','F_S']:
        quantity[arm]={}
        for kind,test in [('best','fixed'),('best','random'),('final','random')]:
            quantity[arm][kind+'_'+test]=ci([json.loads((R/'quantity_control'/f'{arm}_seed{s}_{kind}'/(test+'.json')).read_text())['success'] for s in range(5)])
    effect={
        'F_S-B':paired(quantity['F_S']['best_random']['values'],summary['B']['random']['values']),
        'D-F_S':paired(summary['D']['random']['values'],quantity['F_S']['best_random']['values']),
        'F_S-F':paired(quantity['F_S']['best_random']['values'],summary['F']['random']['values'])}
    ordered=sorted(effect,key=lambda k:effect[k]['paired_t_p']);previous=0
    for i,k in enumerate(ordered):previous=max(previous,min(1.,effect[k]['paired_t_p']*(3-i)));effect[k]['holm_p']=previous
    result['same_S_expert_quantity_control']=quantity;result['same_S_expert_effects']=effect
    (R/'summary.json').write_text(json.dumps(result,indent=2))
    with (R/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figdir=R/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,test,title in zip(axes,['random','artificial_mean','natural_severe'],['Random stations','Artificial recovery','Natural severe deviation']):
        for i,arm in enumerate(ARMS):
            d=summary[arm][test];v=np.asarray(d['values']);ax.scatter(np.full(5,i)+np.linspace(-.1,.1,5),100*v,s=14)
            ax.errorbar(i,100*d['mean'],yerr=100*np.array([[d['mean']-d['ci95'][0]],[d['ci95'][1]-d['mean']]]),fmt='o',color='black',capsize=3)
        ax.set_xticks(range(7),['A','B','C','D','E','F','D−pose']);ax.set_ylim(-5,105);ax.set_title(title);ax.set_ylabel('Success (%)')
    fig.tight_layout();fig.savefig(figdir/'paired_bc_comparison.png',dpi=180);fig.savefig(figdir/'paired_bc_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(random={a:summary[a]['random'] for a in ARMS},paired={k:v['random'] for k,v in comparisons.items()},gate=gate)),flush=True)

if __name__=='__main__':main()
