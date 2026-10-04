"""All four formal arms, paired training-seed statistics, no test selection."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiments.phase14_2_recovery_bc.analyze import ci,paired
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc'
ARMS=['A','B','C','D'];TESTS=['fixed','random','ee_offset','contact_loss','door_regression','stagnation','natural_severe']
def main():
    if not (R/'experiment_complete.json').exists():raise RuntimeError('Formal experiments incomplete')
    selection=json.loads((R/'selection.json').read_text());summary={};rows=[]
    for arm in ARMS:
        summary[arm]={}
        for test in TESTS:
            ds=[json.loads((R/'test'/f'{arm}_seed{i}'/(test+'.json')).read_text()) for i in range(5)]
            metrics=['success'] if test in ['fixed','random'] else ['success','reapproach','recontact','regrasp','progress_resumed']
            summary[arm][test]={k:ci([d[k] for d in ds]) for k in metrics}
            for seed,d in enumerate(ds):rows.append(dict(arm=arm,seed=seed,test=test,success=d['success'],episodes=d['episodes'],
                  selected_update=selection[f'{arm}_seed{seed}']['step'],elapsed_s=d['elapsed_s'],inference=d.get('inference')))
        summary[arm]['artificial_macro']=ci([np.mean([json.loads((R/'test'/f'{arm}_seed{i}'/(t+'.json')).read_text())['success'] for t in TESTS[2:6]]) for i in range(5)])
    comparisons={}
    for name,a,b in [('D-B','D','B'),('C-A','C','A'),('D-C','D','C'),('B-A','B','A')]:
        comparisons[name]={t:paired(summary[a][t]['success']['values'],summary[b][t]['success']['values']) for t in ['random','fixed','natural_severe']}
        comparisons[name]['artificial_macro']=paired(summary[a]['artificial_macro']['values'],summary[b]['artificial_macro']['values'])
    order=sorted(comparisons,key=lambda k:comparisons[k]['random']['paired_t_p']);previous=0
    for i,k in enumerate(order):previous=max(previous,min(1,comparisons[k]['random']['paired_t_p']*(4-i)));comparisons[k]['random']['holm_p']=previous
    gate=dict(random_over80=summary['D']['random']['success']['mean']>.8,paired_model_gain=comparisons['D-B']['random']['ci95'][0]>0 and comparisons['D-B']['random']['holm_p']<.05,
       seed_sd_below10=summary['D']['random']['success']['sd']<.1,artificial_improved=comparisons['D-B']['artificial_macro']['ci95'][0]>0,
       natural_substantial=comparisons['D-B']['natural_severe']['mean']>.1 and comparisons['D-B']['natural_severe']['ci95'][0]>0,
       fixed_retained=summary['D']['fixed']['success']['mean']>=max(summary[a]['fixed']['success']['mean'] for a in ['A','B'])-.05)
    gate['candidate_qualified']=all(gate.values());gate['rl_trained']=False;gate['lwd_divl_ready']=False
    starts={}
    for case in TESTS[2:]:
        arr=[dict(np.load(R/'test'/f'{a}_seed{s}'/(case+'.initial.npz'))) for a in ARMS for s in range(5)]
        robots=np.stack([d['robot'] for d in arr]);states=np.stack([d['state'] for d in arr]);contacts=states[:,:,9:11]>.5
        starts[case]={'max_eef_span_m':float(np.linalg.norm(np.ptp(robots[:,:,18:21],axis=0),axis=-1).max()),
          'max_joint_span_rad':float(np.ptp(robots[:,:,:9],axis=0).max()),'contacts_consistent_fraction':float(np.mean((contacts==contacts[:1]).all((0,2))))}
    d=dict(summary=summary,paired=comparisons,anchor_gate=gate,physical_starts=starts,selection=selection,
         uncertainty='Student-t95CI over5training seeds; pairedt and exactsignflip; n5exacttwo-sided p>=.0625',rl_updates=0)
    (R/'summary.json').write_text(json.dumps(d,indent=2))
    with (R/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='inference']);w.writeheader();w.writerows([{k:v for k,v in row.items() if k!='inference'} for row in rows])
    figdir=R/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,t,title in zip(axes,['random','artificial_macro','natural_severe'],['Random station','Artificial recovery','Natural severe deviation']):
        for i,arm in enumerate(ARMS):
            s=summary[arm][t] if t=='artificial_macro' else summary[arm][t]['success']
            ax.scatter(i+np.linspace(-.1,.1,5),100*np.array(s['values']),s=18)
            ax.errorbar(i,100*s['mean'],yerr=np.array([[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]])*100,fmt='o',color='black',capsize=3)
        ax.set_xticks(range(4),ARMS);ax.set_ylim(-5,105);ax.set_title(title);ax.set_ylabel('Success (%)')
    fig.tight_layout();fig.savefig(figdir/'model_comparison.png',dpi=180);fig.savefig(figdir/'model_comparison.pdf');plt.close(fig)
    print(json.dumps({'random':{a:summary[a]['random']['success'] for a in ARMS},'gate':gate}),flush=True)
if __name__=='__main__':main()
