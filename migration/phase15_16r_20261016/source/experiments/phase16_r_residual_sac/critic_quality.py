"""Held-out Bellman diagnostics and explicitly unverified action rankings."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import h5py,numpy as np,torch
from algorithms.residual_sac.sac import SAC,compose
from experiments.phase16_r_residual_sac.common import ROOT,RES,save,sha,utc,protocol,checkpoint

def main():
    p=protocol();torch.set_num_threads(4);rows=[];ranking=[]
    for arm in ['C','D']:
        for seed in range(3):
            path=ROOT/f'checkpoints/phase16_r/{arm}_seed{seed}/step50000.pt'
            source=RES/f'evaluation/{arm}_seed{seed}/step50000/random_sampled.h5'
            if not path.exists() or not source.exists():continue
            s=SAC(seed,p['residual']['small_bound'],'cpu',p['sac']);s.restore(torch.load(path,map_location='cpu',weights_only=False))
            meta=torch.load(checkpoint(seed),map_location='cpu',weights_only=False);mean=np.asarray(meta['mean'],np.float32);std=np.asarray(meta['std'],np.float32)
            with h5py.File(source) as h:
                raw=h['raw'][:];base=h['base'][:];chunk=h['chunk'][:];cur=h['cursor'][:];done=h['done'][:]
                ctx=np.concatenate((2/np.pi*np.arctan((raw-mean)/std),base,chunk.reshape(*raw.shape[:2],70),np.eye(5,dtype=np.float32)[cur]),-1).astype(np.float32)
                # Fixed lexicographic every seventh tick, all clones. No outcome selection.
                mask=np.zeros(raw.shape[:2],bool);mask[:-1:7]=True
                if 'scored' in h:mask&=h['scored'][:].astype(bool)
                ti,ei=np.where(mask);ti=ti[:2048];ei=ei[:2048]
                b={k:torch.as_tensor(v) for k,v in dict(context=ctx[ti,ei],base=base[ti,ei],next_context=ctx[ti+1,ei],next_base=base[ti+1,ei],delta=h['delta'][:][ti,ei],executed=h['executed'][:][ti,ei],reward=h['reward_original' if arm=='C' else 'reward_signed'][:][ti,ei,None],done=done[ti,ei,None].astype(np.float32)).items()}
                b['next_context'][b['done'][:,0].bool()]=0;b['next_base'][b['done'][:,0].bool()]=0
                with torch.no_grad():
                    torch.manual_seed(161234);target=s.target_value(b);q1,q2=s.q(b['context'],b['delta'],b['executed']);td=torch.minimum(q1,q2)-target
                    candidates={};pol=s.policy(b['context'],b['base'],deterministic=True)
                    for name,delta in [('observed',b['delta']),('zero',torch.zeros_like(b['delta'])),('negative_observed',-b['delta']),('actor_mean',pol['delta'])]:
                        _,act,_=compose(b['base'],delta/s.bound,s.bound);a,c=s.q(b['context'],delta,act);candidates[name]=torch.minimum(a,c)[:,0].numpy()
                rows.append(dict(arm=arm,seed=seed,n=len(ti),td_abs_mean=float(td.abs().mean()),td_rms=float(td.square().mean().sqrt()),q1_q2_abs_mean=float((q1-q2).abs().mean()),q_abs_max=float(torch.maximum(q1.abs(),q2.abs()).max()),target_abs_max=float(target.abs().max()),scope='held-out test, one sampled next-policy Bellman target, not physical counterfactual ground truth'))
                names=list(candidates);scores=np.stack(list(candidates.values()),-1);order=np.argsort(-scores,axis=1)
                for i in range(len(ti)):
                    ranking.append(dict(arm=arm,seed=seed,tick=int(ti[i]),clone=int(ei[i]),best_predicted=names[int(order[i,0])],Q_spread=float(np.ptp(scores[i])),Q_observed=float(scores[i,0]),Q_zero=float(scores[i,1]),Q_negative_observed=float(scores[i,2]),Q_actor_mean=float(scores[i,3]),rank_correctness='unknown: alternatives not physically replayed from this held-out state'))
    for name,values in [('critic_heldout_bellman.csv',rows),('critic_predicted_action_ranking.csv',ranking)]:
        if values:
            with (RES/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    save(RES/'critic_quality.json',dict(utc=utc(),source_sha256=sha(Path(__file__)),rows=rows,action_ranking_correctness='unknown',limitations='Bellman residual tests consistency with the learned target, not calibrated true Q or action ranking correctness. No claim of ranking accuracy from loss decline or unexecuted alternatives.',extra_environment_interactions=0))
    print(json.dumps(dict(trials=len(rows),ranking_states=len(ranking))))
if __name__=='__main__':main()
