"""Standalone scientific figures from real seed-level data."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase11_parameter_generalization.protocol import OUT, ARMS


def main():
    summary=json.loads((OUT/'summary.json').read_text());curves=json.loads((OUT/'curves.json').read_text())
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    colors={'A':'#64748b','B':'#2563eb','C':'#16a34a','D':'#dc2626'}
    labels={'A':'Robot only','B':'+ Angle / Progress','C':'+ Contact','D':'+ Handle XYZ'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm in ARMS:
        for j,name in enumerate(('random','nominal')):
            selected=[curves[f'{arm}_seed{s}'][name] for s in range(5)]
            x=np.array([int(r['env_steps']) for r in selected[0]])/1000
            y=np.array([[float(r['success']) for r in records] for records in selected])
            mean=y.mean(0);ci=2.776445105*y.std(0,ddof=1)/np.sqrt(5)
            axes[j].plot(x,mean,label=labels[arm],color=colors[arm]);axes[j].fill_between(x,mean-ci,mean+ci,color=colors[arm],alpha=.13)
    for ax,title in zip(axes,('Fixed Level2 test distribution','Nominal capability retention')):
        ax.set(xlabel='Additional training interactions (thousands)',ylabel='Stochastic success',title=title,ylim=(-.03,1.05));ax.grid(alpha=.25)
    axes[0].legend(fontsize=9,loc='best');fig.savefig(figures/'success_curves.png',dpi=180);plt.close(fig)
    conditions=['nominal','level1','level2','level3','angle_2p5','angle_5','offset_y_minus_0p01','offset_y_plus_0p01','friction_0p9','friction_1p1']
    matrix=np.array([[summary['generalization'][a]['final'][c]['policy']['mean'] for c in conditions] for a in ARMS])
    fig,ax=plt.subplots(figsize=(12,3.8),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(len(conditions)),['Nominal','Level1','Level2','Level3','Angle +2.5°','Angle +5°','Fixture y −1cm','Fixture y +1cm','Friction ×0.9','Friction ×1.1'],rotation=32,ha='right')
    ax.set_yticks(range(4),[f'{a}: {labels[a]}' for a in ARMS]);ax.set_title('Independent final checkpoint: capped stochastic policy, 5 seed means')
    for i in range(4):
        for j in range(len(conditions)):ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',fontsize=9,color='white' if matrix[i,j]>.6 else 'black')
    fig.colorbar(im,ax=ax,label='Success');fig.savefig(figures/'generalization.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,('auc','final_success','gap'),('Level2 success AUC','Independent final Level2 success','Independent Best − Final')):
        for i,a in enumerate(ARMS):
            item=summary['groups'][a][metric];x=i+np.linspace(-.08,.08,5)
            ax.scatter(x,item['values'],color=colors[a],s=28,alpha=.8)
            ax.errorbar(i,item['mean'],yerr=[[item['mean']-item['ci95'][0]],[item['ci95'][1]-item['mean']]],fmt='o',color='black',capsize=4)
        ax.set_xticks(range(4),ARMS);ax.set_title(title);ax.grid(axis='y',alpha=.25)
        if metric=='gap':ax.axhline(0,color='gray',ls='--',lw=1)
    fig.savefig(figures/'seed_endpoints.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for a in ARMS:
        hist=np.array([summary['quality'][f'{a}_seed{s}']['histogram'] for s in range(5)],dtype=float)
        proportions=hist/hist.sum(1,keepdims=True)
        axes[0].plot(np.linspace(.05,.95,10),proportions.mean(0),marker='o',label=a,color=colors[a])
        exposure=np.array([[summary['curriculum_exposure'][f'{a}_seed{s}'][str(l)]['transitions'] for l in range(4)] for s in range(5)])
        share=exposure.sum(0)/exposure.sum()
        bottom=0.
        for l,value in enumerate(share):axes[1].bar(a,value,bottom=bottom,color=plt.cm.Blues(.25+.2*l),label=f'Level{l}' if a=='A' else None);bottom+=value
    axes[0].set(xlabel='Terminal progress quality score (bin center)',ylabel='Completed-episode fraction',title='Heterogeneous trajectories, uniform replay');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].set(ylabel='Completed transition fraction',title='Realized curriculum exposure');axes[1].legend(fontsize=8)
    fig.savefig(figures/'quality_and_curriculum.png',dpi=180);plt.close(fig)
    print('Saved four scientific figures')
if __name__=='__main__':main()
