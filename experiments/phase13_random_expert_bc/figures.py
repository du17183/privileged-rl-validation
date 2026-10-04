"""Standalone scientific figures from actual seed-level metrics."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'


def save(fig,name):
    folder=R/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    p=json.loads((R/'bc_primary_summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed in range(5):
        rows={r['arm']:r for r in p['per_seed'] if r['model_seed']==seed}
        axes[0].plot([0,1],[rows['A']['success'],rows['B']['success']],marker='o',alpha=.8,label=f'seed {seed}')
    axes[0].set(xticks=[0,1],xticklabels=['Robot-only BC','BC + environment'],ylabel='Randomized success',ylim=(0,1),title='Matched five training seeds')
    axes[0].legend(fontsize=8)
    for i,arm in enumerate(['A','B']):
        v=p['summary'][arm]['success'];axes[1].errorbar(i,v['mean'],yerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],fmt='o',capsize=5,color=['#3467ad','#ca5c27'][i])
    axes[1].set(xticks=[0,1],xticklabels=['Robot-only BC','BC + environment'],ylabel='Success mean / Student t 95% CI',ylim=(-.08,1),title='Seed variance remains large')
    axes[1].axhline(.25,color='gray',ls=':',label='Phase12 ~25% (historical, unmatched)');axes[1].legend(fontsize=7)
    fig.tight_layout();save(fig,'bc_paired_success')
    files=sorted((R/'masking').glob('*.json'));values=[json.loads(path.read_text()) for path in files]
    order=['none','none_repeat','angle_only','coherent_angle_progress_remaining','handle_pose','contact','all_environment']
    values=sorted(values,key=lambda r:order.index(r['mask']))
    labels={'none':'No mask','none_repeat':'No mask\nrepeat','angle_only':'Angle',
            'coherent_angle_progress_remaining':'Angle + progress\n+ remaining','handle_pose':'Handle pose',
            'contact':'Contact','all_environment':'All environment'}
    fig,ax=plt.subplots(figsize=(9,4))
    bars=ax.bar(range(len(values)),[r['success'] for r in values],color='#377a86')
    ax.set_xticks(range(len(values)),[labels[r['mask']] for r in values],fontsize=9)
    for bar,r in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.012,f"{r['success']:.1%}",ha='center',fontsize=9)
    ax.set(ylabel='Success',ylim=(0,1),title='Predeclared seed0 masking; 64 matched episodes / condition')
    fig.tight_layout();save(fig,'bc_parameter_masking')
    if (R/'rl_summary.json').exists():
        result=json.loads((R/'rl_summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
        for arm,color,label in [('A','#666666','Frozen BC'),('B','#be4d43','BC + SAC'),('C','#267d62','BC + SAC + KL')]:
            for seed in range(3):
                train=json.loads((R/f'rl/{arm}_seed{seed}/summary.json').read_text())
                x=[v['environment_steps'] for v in train['curves']];y=[v['success'] for v in train['curves']]
                axes[0].plot(x,y,color=color,alpha=.55,label=label if seed==0 else None,marker='.' if arm=='A' else None)
            metric=result['summary'][arm]['final_success'];i=['A','B','C'].index(arm)
            axes[1].errorbar(i,metric['mean'],yerr=[[metric['mean']-metric['ci95'][0]],[metric['ci95'][1]-metric['mean']]],fmt='o',color=color,capsize=5)
            axes[1].scatter([i]*3,[r['final_success'] for r in result['per_seed'] if r['arm']==arm],color=color,alpha=.65)
        axes[0].set(xlabel='Online environment interactions',ylabel='Fixed validation success',ylim=(0,1),title='Three matched RL seeds / arm');axes[0].legend()
        axes[1].set(xticks=[0,1,2],xticklabels=['Frozen BC','BC + SAC','BC + SAC + KL'],ylabel='Heldout final success / t 95% CI',title='Untouched heldout cohort; n=3')
        fig.tight_layout();save(fig,'rl_stability')
    print(R/'figures')


if __name__=='__main__':main()
