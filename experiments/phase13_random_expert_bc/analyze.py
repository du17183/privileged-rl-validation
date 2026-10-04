"""Seed-level paired inference, honest gates, and offline input sensitivity."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
import torch
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs


def wilson(k,n):
    z=1.959963984540054;x=k/n;den=1+z*z/n
    center=(x+z*z/(2*n))/den
    delta=z*np.sqrt(x*(1-x)/n+z*z/(4*n*n))/den
    return [float(center-delta),float(center+delta)]


def estimate(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=x.mean();sd=x.std(ddof=1) if n>1 else 0
    delta=stats.t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else float('nan')
    return dict(mean=float(mean),seed_sd=float(sd),n=n,ci95=[float(mean-delta),float(mean+delta)])


def paired(b,a):
    delta=np.asarray(b)-np.asarray(a)
    result=estimate(delta)
    result['per_seed_difference']=delta.tolist()
    if np.std(delta)==0:
        result['paired_t_p']=0. if delta[0]!=0 else 1.
    else:result['paired_t_p']=float(stats.ttest_rel(b,a).pvalue)
    distribution=[abs(np.mean(delta*np.asarray(signs))) for signs in itertools.product([-1,1],repeat=len(delta))]
    result['exact_sign_flip_p']=float(np.mean(np.asarray(distribution)>=abs(delta.mean())-1e-12))
    result['positive_seeds']=int((delta>0).sum())
    return result


def primary(root):
    root=Path(root);rows=[]
    for seed in range(5):
        for arm in ['A','B']:
            row=json.loads((root/f'heldout/{arm}_seed{seed}_best.json').read_text())
            if row['episodes']!=128 or row['stochastic'] or row['mask']!='none':raise RuntimeError('Primary protocol mismatch')
            rows.append(row)
    grouped={arm:[r for r in rows if r['arm']==arm] for arm in ['A','B']}
    summary={arm:{metric:estimate([r[metric] for r in grouped[arm]]) for metric in
                       ['success','mean_max_angle_rad','mean_final_angle_rad','contact_success']} for arm in grouped}
    differences={metric:paired([r[metric] for r in grouped['B']],[r[metric] for r in grouped['A']])
                 for metric in ['success','mean_max_angle_rad','mean_final_angle_rad','contact_success']}
    d=differences['success']
    gate=bool(summary['B']['success']['mean']>.5 and d['mean']>=.1 and d['ci95'][0]>0 and d['positive_seeds']>=4)
    result=dict(per_seed=rows,summary=summary,paired=differences,anchor_gate_passed=gate,
                gate='B >50%; B-A >=10pp; paired t 95% CI >0; >=4 positive seeds',
                inference_note='n=5; t intervals assume independent approximately normal seed effects. Exact two-sided sign-flip p has minimum .0625; do not claim exact p<.05.',
                evaluation_interactions=sum(r['interactions'] for r in rows))
    (root/'bc_primary_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summary=summary,paired_success=d,anchor_gate_passed=gate)),flush=True)
    return result


@torch.no_grad()
def sensitivity(data,checkpoints,output):
    arrays=dict(np.load(Path(data)/'test.npz'))
    eligible=np.flatnonzero((arrays['environment'][:,0]<.15)&(arrays['environment'][:,2]<.15))
    rng=np.random.default_rng(13701);ids=rng.choice(eligible,min(256,len(eligible)),replace=False)
    robot=torch.tensor(arrays['robot'][ids]);context=torch.tensor(arrays['environment'][ids])
    rows=[]
    for seed in range(5):
        for arm in ['A','B']:
            policy,mean,std,value=load_checkpoint(Path(checkpoints)/f'{arm}_seed{seed}_best.pt','cpu')
            base=policy(inputs(robot,context,mean,std,arm))
            predictions=[];coherent=[]
            for degree in np.linspace(0,5,11):
                modified=context.clone();modified[:,0]=np.deg2rad(degree)
                predictions.append(policy(inputs(robot,modified,mean,std,arm)))
                start=(context[:,0]-context[:,2]*context[:,1])/(1-context[:,2]).clamp_min(1e-6)
                modified[:,2]=((modified[:,0]-start)/(modified[:,1]-start).clamp_min(1e-6)).clamp(0,1)
                modified[:,3]=modified[:,1]-modified[:,0]
                coherent.append(policy(inputs(robot,modified,mean,std,arm)))
            raw=torch.stack(predictions);consistent=torch.stack(coherent)
            rows.append(dict(seed=seed,arm=arm,states=len(ids),
                             raw_angle_rms_change=float((raw-base[None]).square().mean().sqrt()),
                             coherent_angle_rms_change=float((consistent-base[None]).square().mean().sqrt()),
                             raw_angle_max_action_range=float((raw.max(0).values-raw.min(0).values).abs().max()),
                             coherent_angle_max_action_range=float((consistent.max(0).values-consistent.min(0).values).abs().max())))
    result=dict(rows=rows,degrees=np.linspace(0,5,11).tolist(),state_indices=ids.tolist(),
                note='Fixed heldout robot states and handle/contact. This counterfactual diagnostic proves numerical parameter use, not physically feasible performance; rollout masking is assessed separately.')
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/phase13_random_expert_bc');p.add_argument('--sensitivity',action='store_true')
    p.add_argument('--data',default='datasets/random_door_expert/split_v1');p.add_argument('--checkpoints',default='checkpoints/phase13_random_expert_bc')
    a=p.parse_args()
    if a.sensitivity:sensitivity(a.data,a.checkpoints,Path(a.root)/'action_sensitivity.json')
    else:primary(a.root)
