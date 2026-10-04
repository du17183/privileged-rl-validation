"""Standalone research figures; actual stochastic policy curves are primary."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase10_quality_lwd.protocol import ARMS
from experiments.phase10_quality_lwd.analyze import read
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
COLORS=['#555555','#ad7c35','#436ab0','#73adce','#009e73','#71b47c','#b45686']


def main():
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    summary=json.loads((OUT/'summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for arm,color in zip(ARMS,COLORS):
        all_rows=[read(OUT/f'eval_P10{arm}_seed{s}_policy.csv') for s in range(5)]
        x=np.array([int(r['env_steps']) for r in all_rows[0]])/1000
        for axis,key in zip(axes,('success','return')):
            ys=np.array([[float(r[key]) for r in rows] for rows in all_rows])
            mean=ys.mean(0);half=2.7764451051977987*ys.std(0,ddof=1)/np.sqrt(5)
            axis.plot(x,mean,label=arm,color=color)
            axis.fill_between(x,mean-half,mean+half,color=color,alpha=.07)
    axes[0].set(ylabel='Executed stochastic success rate',xlabel='Additional training interactions (k)',ylim=(0,1.05))
    axes[1].set(ylabel='Episode return (unchanged progress reward)',xlabel='Additional training interactions (k)')
    axes[0].legend(ncol=4,loc='lower left');fig.suptitle('Phase 10: paired continuation from Phase 9 C final, five seeds')
    fig.tight_layout();fig.savefig(figures/'success_reward_curves.png');plt.close(fig)
    names=list(ARMS);x=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric,label in [(axes[0],'policy_auc','Success AUC'),(axes[1],'policy_final','Independent final stochastic success')]:
        means=[summary[a][metric]['mean'] for a in names]
        errors=[summary[a][metric]['mean']-summary[a][metric]['ci95'][0] for a in names]
        ax.bar(x,means,color=COLORS);ax.errorbar(x,means,yerr=errors,fmt='none',color='#333333',capsize=3)
        ax.set(xticks=x,xticklabels=names,ylabel=label,ylim=(0,1.1))
        for i,a in enumerate(names):
            ys=summary[a][metric]['per_seed'];ax.scatter(np.linspace(i-.12,i+.12,5),ys,color='#333333',s=12,zorder=3)
    fig.suptitle('Seed means and t95 CI; raw CI is not clipped');fig.tight_layout();fig.savefig(figures/'auc_endpoints.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    bottom=np.zeros(len(names))
    for key,color,label in [('expert','#436ab0','Expert'),('online_success','#009e73','Completed online success'),('online_failure','#b45686','Completed online failure'),('online_partial','#ad7c35','Outcome not yet known at draw')]:
        values=np.array([summary[a][f'sample_{key}_fraction']['mean'] for a in names])
        ax.bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
    ax.set(xticks=x,xticklabels=names,ylabel='Actual SAC replay draw fraction',ylim=(0,1))
    ax.legend(ncol=2,loc='upper right');fig.tight_layout();fig.savefig(figures/'sampling_distribution.png');plt.close(fig)
    tests=json.loads((OUT/'heldout_summary.json').read_text())
    conditions=['nominal','angle_2p5','angle_5','handle_y_minus_1cm','handle_y_plus_1cm','friction_0p9','friction_1p1']
    arms=['REFC','B','D10','E']
    matrix=np.array([[next(t['success']['mean'] for t in tests if t['arm']==a and t['condition']==c and t['checkpoint']=='final' and t['mode']=='policy') for c in conditions] for a in arms])
    fig,ax=plt.subplots(figsize=(11,3.4))
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set(yticks=range(4),yticklabels=arms,xticks=range(7),xticklabels=['Nominal','Angle +2.5deg','Angle +5deg','Fixture y -1cm','Fixture y +1cm','Friction x0.9','Friction x1.1'])
    for i in range(4):
        for j in range(7):ax.text(j,i,f'{matrix[i,j]*100:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.6 else '#333333')
    fig.colorbar(im,ax=ax,label='Final stochastic success');fig.tight_layout();fig.savefig(figures/'generalization.png');plt.close(fig)

if __name__=='__main__':main()
