"""Compare saved Q with observed reward/soft returns of completed rollouts."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,h5py,numpy as np
from algorithms.residual_sac.sac import SAC
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol,checkpoint

def main():
    P=protocol();torch.set_num_threads(4);rows=[];states=[]
    for arm in ['C','D']:
        for seed in range(3):
            model=ROOT/f'checkpoints/phase16_r/{arm}_seed{seed}/step50000.pt';source=RES/f'evaluation/{arm}_seed{seed}/step50000/random_sampled.h5'
            if not model.exists() or not source.exists():continue
            assert (source.parent/'complete.json').exists(),'Final independent rollout not closed'
            s=SAC(seed,P['residual']['small_bound'],'cpu',P['sac']);s.restore(torch.load(model,map_location='cpu',weights_only=False));alpha=float(s.log_alpha.exp());gamma=P['sac']['gamma']
            normal=torch.load(checkpoint(seed),map_location='cpu',weights_only=False);mean=np.asarray(normal['mean'],np.float32);std=np.asarray(normal['std'],np.float32)
            with h5py.File(source) as h:
                raw=h['raw'][:];base=h['base'][:];chunk=h['chunk'][:];cur=h['cursor'][:];done=h['done'][:].astype(bool);scored=h['scored'][:].astype(bool);episode=h['episode'][:];age=h['age'][:];delta=h['delta'][:];executed=h['executed'][:];reward=h['reward_original' if arm=='C' else 'reward_signed'][:];logp=h['logp'][:].reshape(done.shape)
            context=np.concatenate((2/np.pi*np.arctan((raw-mean)/std),base,chunk.reshape(*raw.shape[:2],70),np.eye(5,dtype=np.float32)[cur]),-1).astype(np.float32)
            empirical_reward=np.full(done.shape,np.nan);empirical_soft=np.full(done.shape,np.nan);closed=np.zeros(done.shape,bool);closed_episodes=0;censored_episodes=0
            for clone in range(done.shape[1]):
                for identifier in np.unique(episode[:,clone]):
                    indexes=np.flatnonzero((episode[:,clone]==identifier)&scored[:,clone])
                    if not len(indexes):continue
                    # Include all completed scored successes, timeouts and unsafe
                    # episodes; retain right-censored scope without inventing G.
                    if age[indexes[0],clone]!=0 or not done[indexes[-1],clone] or not np.all(np.diff(indexes)==1):censored_episodes+=1;continue
                    reward_return=0.;soft_return=0.;next_logp=0.;closed_episodes+=1
                    for tick in indexes[::-1]:
                        reward_return=float(reward[tick,clone])+gamma*reward_return
                        soft_return=float(reward[tick,clone])+gamma*(soft_return-alpha*next_logp)
                        empirical_reward[tick,clone]=reward_return;empirical_soft[tick,clone]=soft_return;next_logp=float(logp[tick,clone])
                    closed[indexes,clone]=True
            mask=np.zeros(done.shape,bool);mask[::7]=True;mask&=closed
            ticks,clones=np.where(mask);ticks=ticks[:2048];clones=clones[:2048]
            if not len(ticks):continue
            with torch.no_grad():
                q1,q2=s.q(torch.as_tensor(context[ticks,clones]),torch.as_tensor(delta[ticks,clones]),torch.as_tensor(executed[ticks,clones]));predicted=torch.minimum(q1,q2)[:,0].numpy()
            gr=empirical_reward[ticks,clones];gs=empirical_soft[ticks,clones];error=predicted-gs
            finite=bool(np.isfinite(predicted).all() and np.isfinite(gs).all())
            rows.append(dict(arm=arm,seed=seed,n=len(ticks),closed_scored_episodes=closed_episodes,right_censored_scored_episodes=censored_episodes,mean_Q=float(predicted.mean()),mean_observed_reward_return=float(gr.mean()),mean_observed_soft_return=float(gs.mean()),Q_minus_observed_soft_bias=float(error.mean()),absolute_error_mean=float(np.mean(abs(error))),error_RMS=float(np.sqrt(np.mean(error**2))),absolute_error_p50=float(np.quantile(abs(error),.5)),absolute_error_p95=float(np.quantile(abs(error),.95)),alpha=alpha,gamma=gamma,finite=finite,checkpoint_sha256=sha(model),trace_sha256=sha(source),scope='one realized complete sampled-policy trajectory per state; correlated state samples, not calibrated conditional expected Q or same-state counterfactual ranking'))
            for i in range(len(ticks)):states.append(dict(arm=arm,seed=seed,tick=int(ticks[i]),clone=int(clones[i]),episode=int(episode[ticks[i],clones[i]]),age=int(age[ticks[i],clones[i]]),Q_min=float(predicted[i]),observed_reward_return=float(gr[i]),observed_soft_return=float(gs[i]),Q_minus_observed_soft=float(error[i])))
    for name,data in [('critic_observed_return.csv',rows),('critic_observed_return_states.csv',states)]:
        if data:
            with (RES/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save(RES/'critic_observed_return.json',dict(utc=utc(),source_sha256=sha(Path(__file__)),rows=rows,selection='every seventh absolute tick, all clones, first2048 qualifying states lexicographically; all closed scored outcomes included, open episodes explicitly excluded as censored',return_definition='G_reward[t]=r[t]+gamma*G_reward[t+1]; G_soft[t]=r[t]+gamma*(G_soft[t+1]-alpha*logp_unit[t+1]); terminal has no future term; current-action entropy excluded as in Q target',limitations='single realized return contains trajectory noise and cannot alone calibrate expected Q. Alternative actions were not physically replayed, action-ranking correctness remains unknown.',physical_interactions_added=0,training_updates_added=0))
    print(json.dumps(dict(trials=len(rows),rows=rows)))
if __name__=='__main__':main()
