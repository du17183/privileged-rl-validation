"""Postprocessing only: quantify saved physical actions, costs and learning curves."""
import sys,json,csv,math
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol

def read(p):return json.loads(p.read_text()) if p.exists() else None
def csvwrite(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=protocol();rows=[];strata=[];action_rows=[];online=[];auc=[];calls=[]
    files=[]
    for seed in range(5):
        for arm in ['A','B']:
            for scope in ['random','fixed','validation']:
                files.append((RES/f'baseline/seed{seed}/{arm}_{scope}',arm,seed,0,scope,'base_flow'))
    for arm in ['C','D']:
        for seed in range(3):
            for step in p['evaluation']['nodes']:
                for scope in ['random','fixed','validation']:
                    for dist in ['sampled','mean']:
                        files.append((RES/f'evaluation/{arm}_seed{seed}/step{step}/{scope}_{dist}',arm,seed,step,scope,dist))
            for scope in ['random','fixed']:
                for dist in ['sampled','mean']:
                    files.append((RES/f'evaluation/{arm}_seed{seed}/best_test/{scope}_{dist}',arm,seed,-1,scope,dist))
            files.append((RES/f'train/{arm}_seed{seed}/training',arm,seed,50000,'training','sampled'))
    for path,arm,seed,step,scope,dist in files:
        meta=read(path.with_suffix('.json'))
        if scope!='training' and meta is None:continue
        if scope=='training' and not (path.parent/'complete.json').exists() and not (path.parent/'stopped.json').exists():continue
        if path.with_suffix('.csv').exists():
            with path.with_suffix('.csv').open() as f:episodes=list(csv.DictReader(f))
            if episodes and scope!='training':
                ticks=np.asarray([int(x['ticks']) for x in episodes]);succ=np.asarray([int(x['success']) for x in episodes]);angles=np.asarray([float(x['actual_angle_before_policy']) for x in episodes]);asked=np.asarray([float(x['requested_angle']) for x in episodes]);offset=np.asarray([max(abs(float(x[k])) for k in ['offset_x','offset_y','offset_z']) for x in episodes]);angle30=np.asarray([float(x['actual_angle_tick30']) if x['actual_angle_tick30'] else np.nan for x in episodes])
                rows.append(dict(arm=arm,seed=seed,step=step,scope=scope,distribution=dist,n=len(episodes),success=float(succ.mean()),mean_ticks=float(ticks.mean()),mean_sim_seconds=float(ticks.mean()/60),before_angle_mean=float(angles.mean()),tick30_angle_mean=float(np.nanmean(angle30)),wall_s=meta['elapsed_s'],physical_interactions=meta['physical_interactions'],automatic_resets=meta['automatic_resets']))
                for dimension,values,bins in [('requested_angle_deg',np.degrees(asked),[(0,2.5),(2.5,5.001)]),('actual_before_angle_deg',np.degrees(angles),[(0,2.5),(2.5,5.001)]),('fixture_linf_mm',offset*1000,[(0,5),(5,10.001)])]:
                    for low,high in bins:
                        mask=(values>=low)&(values<high);n=int(mask.sum())
                        strata.append(dict(arm=arm,seed=seed,step=step,scope=scope,distribution=dist,dimension=dimension,lower=low,upper=high,n=n,success=float(succ[mask].mean()) if n else None,mean_ticks=float(ticks[mask].mean()) if n else None))
        if path.with_suffix('.h5').exists():
            with h5py.File(path.with_suffix('.h5')) as h:
                raw=h['raw'][:];base=h['base'][:];delta=h['delta'][:];executed=h['executed'][:];clipped=h['clip'][:]
                masks={'all':np.ones(raw.shape[:2],bool),'bilateral_contact':(raw[:,:,30]>.5)&(raw[:,:,31]>.5),'no_bilateral_contact':~((raw[:,:,30]>.5)&(raw[:,:,31]>.5))}
                if 'scored' in h:
                    for k in masks:masks[k]&=h['scored'][:].astype(bool)
                for stage,mask in masks.items():
                    if not np.any(mask):continue
                    for name,data in [('base',base),('residual',delta),('executed',executed),('actual_change',executed[:,:,:6]-base[:,:,:6])]:
                        a=data[mask]
                        for axis in range(a.shape[-1]):
                            x=a[:,axis]
                            action_rows.append(dict(arm=arm,seed=seed,step=step,scope=scope,distribution=dist,contact_scope=stage,action_kind=name,axis=axis,n=len(x),mean=float(x.mean()),std=float(x.std()),rms=float(np.sqrt(np.mean(x*x))),abs_p50=float(np.quantile(abs(x),.5)),abs_p95=float(np.quantile(abs(x),.95)),abs_p99=float(np.quantile(abs(x),.99)),abs_max=float(abs(x).max()),clip_fraction=float(clipped[mask].mean())))
        if meta:
            for x in meta['calls']:calls.append(dict(arm=arm,seed=seed,scope=scope,distribution=dist,**x))
    for arm in ['C','D']:
        for seed in range(3):
            folder=RES/f'train/{arm}_seed{seed}';ep=folder/'online_episodes.csv'
            if ep.exists():
                with ep.open() as f:values=list(csv.DictReader(f))
                for lo,hi in [(0,10000),(10000,25008),(25008,50000)]:
                    selected=[x for x in values if lo<int(x['training_interactions'])<=hi]
                    n=len(selected);s=sum(int(x['success']) for x in selected)
                    online.append(dict(arm=arm,seed=seed,lo_interactions=lo,hi_interactions=hi,completed_episodes=n,successes=s,failures=n-s,completion_success_rate=s/n if n else None))
            for dist in ['sampled','mean']:
                curve=[(step,read(RES/f'evaluation/{arm}_seed{seed}/step{step}/validation_{dist}.json')) for step in p['evaluation']['nodes']]
                curve=[(x,y['success']) for x,y in curve if y]
                if len(curve)>1:
                    x,y=np.asarray(curve).T
                    auc.append(dict(arm=arm,seed=seed,distribution=dist,points=len(curve),last_interaction=int(x[-1]),normalized_auc=float(np.trapz(y,x)/(x[-1]-x[0])),validation_only=True))
    csvwrite(RES/'episode_scope_metrics.csv',rows);csvwrite(RES/'random_strata.csv',strata);csvwrite(RES/'action_distributions.csv',action_rows);csvwrite(RES/'online_success_segments.csv',online);csvwrite(RES/'validation_auc.csv',auc)
    timing=[]
    for seed in range(5):
        for arm in ['A','B']:
            for scope in ['random','fixed','validation']:
                values=[x for x in calls if x['seed']==seed and x['arm']==arm and x['scope']==scope]
                for batch in sorted({x['batch'] for x in values}):
                    for clock in ['latency_s','service_s','rpc_s']:
                        arr=np.array([x[clock] for x in values if x['batch']==batch]);timing.append(dict(arm=arm,seed=seed,scope=scope,batch=batch,clock=clock,n=len(arr),total_s=float(arr.sum()),p50_s=float(np.quantile(arr,.5)),p95_s=float(np.quantile(arr,.95)),p99_s=float(np.quantile(arr,.99)),deployment_exclusive=False))
    csvwrite(RES/'inference_timing_by_scope_batch.csv',timing)
    figdir=RES/'figures';figdir.mkdir(exist_ok=True)
    if auc:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for arm,color in [('C','#3976af'),('D','#d26a27')]:
            for seed in range(3):
                pts=[(x['step'],x['success']) for x in rows if x['arm']==arm and x['seed']==seed and x['scope']=='validation' and x['distribution']=='sampled'];pts=sorted(pts)
                if pts:xx,yy=np.asarray(pts).T;ax.plot(xx,100*yy,marker='o',color=color,alpha=.65,label=f'{arm} seed {seed}')
        ax.set(xlabel='Training physical interactions',ylabel='Sampled validation success (%)',ylim=(0,100));ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(figdir/f'validation_curves.{ext}',dpi=180)
        plt.close(fig)
    save(RES/'postprocess_evidence.json',dict(utc=utc(),source_sha256=sha(Path(__file__)),episode_scopes=len(rows),action_strata=len(action_rows),online_segments=online,auc=auc,scope='saved completed physical rollouts; per-axis moments and quantiles from individual actual tick commands; no pool percentile averages',collision_observation='arm/environment collision not measured by this task interface',action_ranking='critic rankings require a held-out physical or Bellman validation; loss decrease alone does not prove correct ranking'))
    print(json.dumps(dict(scopes=len(rows),action_rows=len(action_rows),auc=len(auc))))

if __name__=='__main__':main()
