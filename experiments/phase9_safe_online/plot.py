"""Standalone scientific figures; shaded bands are five-seed t95 intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.phase9_safe_online.analyze import read
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'
FIG=OUT/'figures'
COLORS=dict(A='#333333',B='#dd7c18',C='#247ab0',D='#199666',CANN='#9352aa',D100='#b24751',D30='#6f7a16')
plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})

def band(axis,x,data,label,probability=False):
    data=np.asarray(data,dtype=float)
    mean=data.mean(0);half=2.7764451051977987*data.std(0,ddof=1)/np.sqrt(5)
    axis.plot(x,mean,label=label,color=COLORS[label],linewidth=1.8)
    lo,hi=mean-half,mean+half
    if probability:lo,hi=np.maximum(0,lo),np.minimum(1,hi)
    axis.fill_between(x,lo,hi,color=COLORS[label],alpha=.13)

def main():
    FIG.mkdir(exist_ok=True)
    summary=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for index,arms in enumerate((('A','B','C','D'),('C','CANN','D100','D30'))):
        for mode,col in [('policy',0),('deterministic',1)]:
            ax=axes[index,col]
            for arm in arms:
                rows=[read(OUT/f'eval_P9{arm}_seed{s}_{mode}.csv') for s in range(5)]
                x=[int(r['env_steps'])/1000 for r in rows[0]]
                band(ax,x,[[float(r['success']) for r in v] for v in rows],arm,True)
            ax.set_ylim(-.025,1.025);ax.set_ylabel('Success rate');ax.set_title(('Primary' if index==0 else 'Secondary')+f' / {mode}')
            ax.legend(loc='lower right',ncol=2);ax.grid(alpha=.2)
            if index==1:ax.set_xlabel('Additional Phase 9 training interactions (thousands)')
    fig.suptitle('Accepted policies / five paired seeds / t95 CI')
    fig.tight_layout();fig.savefig(FIG/'success_curves.png');fig.savefig(FIG/'success_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for arm in ('A','B','C','D'):
        rows=[read(OUT/f'eval_P9{arm}_seed{s}_policy.csv') for s in range(5)]
        x=[int(r['env_steps'])/1000 for r in rows[0]]
        for ax,key in zip(axes,('online_successes','anchor_kl','effective_std')):
            band(ax,x,[[float(r[key]) for r in v] for v in rows],arm)
    for ax,title,ylabel in zip(axes,('New exploration successes','Accepted policy drift','Executed pre-tanh std'),('Completed successes (mean / seed)','KL(new || frozen anchor)','Standard deviation')):
        ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel('Additional training interactions (thousands)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(FIG/'online_and_distribution.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    arms=list(summary)
    for ax,mode in zip(axes,('policy','det')):
        x=np.arange(len(arms));w=.24
        for index,key in enumerate((mode+'_cap_matched_anchor',mode+'_best',mode+'_final')):
            mean=np.array([summary[a][key]['mean'] for a in arms]);sd=np.array([summary[a][key]['sd'] for a in arms])
            ax.bar(x+(index-1)*w,mean,w,label=key.removeprefix(mode+'_'),yerr=sd*2.7764451051977987/np.sqrt(5),capsize=2)
        ax.set_xticks(x,arms);ax.set_ylim(0,1.12);ax.set_title(f'Independent nominal {mode}');ax.set_ylabel('Success rate');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'independent_checkpoints.png');plt.close(fig)
    held=json.loads((OUT/'heldout_summary.json').read_text())
    conditions=['nominal','angle_2p5','angle_5','handle_y_minus_1cm','handle_y_plus_1cm']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,mode in zip(axes,('policy','deterministic')):
        for arm in ('A','B','C','D'):
            rows=[next(r for r in held if r['arm']==arm and r['checkpoint']=='final' and r['condition']==c and r['mode']==mode) for c in conditions]
            ax.errorbar(np.arange(5),[r['success']['mean'] for r in rows],
                yerr=[r['success']['sd']*2.7764451051977987/np.sqrt(5) for r in rows],label=arm,color=COLORS[arm],marker='o',capsize=2)
        ax.set_xticks(np.arange(5),['Nominal','Door +2.5 deg','Door +5 deg','Handle -1 cm','Handle +1 cm'],rotation=18)
        ax.set_title('Independent final / '+mode);ax.set_ylim(-.025,1.1);ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Success rate / t95 CI');fig.tight_layout();fig.savefig(FIG/'generalization.png');plt.close(fig)

if __name__=='__main__':main()
