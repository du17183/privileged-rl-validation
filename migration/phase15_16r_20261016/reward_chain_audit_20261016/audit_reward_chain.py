"""Read-only CPU forensics of already closed Phase16-R data. No updates or Isaac."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, hashlib, csv, math
from pathlib import Path
import numpy as np
import h5py
import torch

ROOT = Path('/DATA/disk1/home/xiaolong/privileged_rl_validation')
sys.path.insert(0, str(ROOT))
from algorithms.residual_sac.sac import SAC
from experiments.phase16_r_residual_sac.common import checkpoint
torch.set_num_threads(2)
P = json.loads((ROOT/'experiments/phase16_r_residual_sac/protocol.json').read_text())
R = ROOT/'results/phase16_r'
OUT = {'scope': 'read-only CPU forensic audit; no simulation, collection, training, signals, or history edits',
       'physical_interactions_added': 0, 'optimizer_updates_added': 0,
       'sources': {}, 'trajectories': [], 'replay_checks': [], 'policy_checks': [], 'initial_pairing': []}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def quant(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x): return {'n': 0}
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.quantile(x,.5)), p95=float(np.quantile(x,.95)), min=float(x.min()), max=float(x.max()))

def maxabs(x): return float(np.max(np.abs(x))) if np.size(x) else 0.

for rel, want in {
    'algorithms/residual_sac/sac.py': '08963008b7b739ea3ff2f80cad76e7a6f98b89e2830a5a994a0f30293bf2fb14',
    'evaluation/phase16_r/runner_v2.py': '476b89199e01f81b86a3f7ff6a59f83b35c18ed792c48361d4d54e670f21b876',
    'evaluation/phase16_r/chunk.py': 'a207615a08c580d9dd9432e2f1df81df8edbee4fc94865c1452712d71a793620',
    'evaluation/phase16_r/env.py': '375e3182408244c552baa079a5b6619ffcac9cc8704406df2d12a7f46bf00887',
    'experiments/phase16_r_residual_sac/protocol.json': '81a4fc5c28dda50349a361b628d2f4044178946dcfbab06010f7180f2be8a91b'} .items():
    got = sha(ROOT/rel); assert got == want, (rel,got)
    OUT['sources'][rel] = got

meta = torch.load(checkpoint(0), map_location='cpu', weights_only=False)
mean, std = np.asarray(meta['mean'],np.float32), np.asarray(meta['std'],np.float32)
del meta

OUT['discount'] = {'control_hz':60, 'gamma':.99, 'effective_steps':100,
    'effective_seconds':100/60, 'half_life_seconds':math.log(.5)/math.log(.99)/60,
    'success_event_integrated_reward':10,
    'weights': {str(t): {str(g):g**t for g in [.99,.997,.999]} for t in [60,100,300,400,500,600]}}

for arm in ['C','D']:
    path = R/f'train/{arm}_seed0/replay_final.npz'
    with np.load(path) as b:
        c, nc, base, nb = [b[k] for k in ['context','next_context','base','next_base']]
        delta, action, done = b['delta'], b['executed'], b['done'][:,0].astype(bool)
        live = ~done[:-16]
        OUT['replay_checks'].append(dict(arm=arm,n=len(c),file_sha256=sha(path),
            context_dim=c.shape[1], next_context_adjacent_error=maxabs(nc[:-16][live]-c[16:][live]),
            next_base_adjacent_error=maxabs(nb[:-16][live]-base[16:][live]),
            terminal_next_context_max=maxabs(nc[done]), terminal_next_base_max=maxabs(nb[done]),
            executed_composition_error=maxabs(action[:,:6]-np.clip(base[:,:6]+delta,-1,1)),
            gripper_equal=bool(np.array_equal(action[:,6],base[:,6])),
            selected_reward_error=maxabs(b['reward']-b['original_reward' if arm=='C' else 'signed_reward']),
            context_base_error=maxabs(c[:,39:46]-base),
            residual_outside_bound=int(np.sum(np.abs(delta)>np.asarray(P['residual']['small_bound'])+1e-7)),
            actor_updates_before_first_completed_episode=None))
    with (R/f'train/{arm}_seed0/updates.csv').open() as f: updates=list(csv.DictReader(f))
    with (R/f'train/{arm}_seed0/online_episodes.csv').open() as f: episodes=list(csv.DictReader(f))
    first_episode=min(int(e['training_interactions']) for e in episodes)
    prior=[u for u in updates if int(u['interactions'])<first_episode]
    OUT['replay_checks'][-1].update(first_completed_episode_interactions=first_episode,
        actor_updates_before_first_completed_episode=int(prior[-1]['updates']) if prior else 0,
        completed_online_episodes=len(episodes), online_successes=sum(int(e['success']) for e in episodes),
        update_log_rows=len(updates), final_optimizer_updates=int(updates[-1]['updates']),
        final_metrics={k:float(v) for k,v in updates[-1].items()},
        max_q_seen=max(float(u['q_abs_max']) for u in updates),
        first_delta_rms=float(updates[0]['delta_rms']),
        last100_delta_rms_mean=float(np.mean([float(u['delta_rms']) for u in updates[-100:]])))

cases=[('A','baseline/seed0/A_random')]
for arm in ['C','D']:
    cases.append((arm,f'train/{arm}_seed0/training'))
    for name in ['validation_sampled','validation_mean','random_sampled','random_mean']:
        cases.append((arm,f'evaluation/{arm}_seed0/step50000/{name}'))

for arm,rel in cases:
    path=R/(rel+'.h5')
    with h5py.File(path,'r') as h:
        a={k:h[k][:] for k in ['raw','next_robot','next_gt','base','delta','executed','age','episode','done','reward_original','reward_signed','logp']}
        a['scored']=h['scored'][:].astype(bool) if 'scored' in h else np.ones(a['done'].shape,bool)
        if '/evaluation/' in str(path):
            a['chunk']=h['chunk'][:];a['cursor']=h['cursor'][:]
    raw, nr, ng = a['raw'],a['next_robot'],a['next_gt']
    dr = 20*(ng[:,:,0]-raw[:,:,26])
    vel=20*np.maximum(ng[:,:,1],0)/60
    reach=(1-np.tanh(5*np.linalg.norm(nr[:,:,18:21]-ng[:,:,2:5],axis=-1)))/60
    grasp=.5*((ng[:,:,9]>0)&(ng[:,:,10]>0))/60
    succ=10*(ng[:,:,0]>1)
    pos=np.maximum(dr,0);neg=np.minimum(dr,0)
    row={'arm':arm,'path':rel,'sha256':sha(path),'steps_including_nonscored':int(a['done'].size),
         'reward_signed_formula_max_error':maxabs(a['reward_signed']-(a['reward_original']-vel+dr)),
         'native_reward_components_max_error':maxabs(a['reward_original']-(reach+grasp+vel+succ)),
         'delta_vector_l2_rms':float(np.sqrt(np.mean(np.sum(a['delta']**2,axis=-1)))),
         'effective_delta_vector_l2_rms':float(np.sqrt(np.mean(np.sum((a['executed'][:,:,:6]-a['base'][:,:,:6])**2,axis=-1)))),
         'residual_bound_fraction_gt90':float(np.mean(np.abs(a['delta'])>.9*np.asarray(P['residual']['small_bound']))),
         'base_at_limit_fraction':float(np.mean(np.abs(a['base'][:,:,:6])>.9999)),
         'episodes':[]}
    components={'original':a['reward_original'],'signed':a['reward_signed'],'reach':reach,'grasp':grasp,'positive_velocity':vel,'net_progress':dr,'positive_progress':pos,'negative_progress':neg,'success_event':succ}
    for clone in range(a['done'].shape[1]):
        for ep in np.unique(a['episode'][:,clone]):
            ix=np.flatnonzero((a['episode'][:,clone]==ep)&a['scored'][:,clone])
            if not len(ix) or a['age'][ix[0],clone]!=0 or not a['done'][ix[-1],clone]:continue
            e={'clone':clone,'episode':int(ep),'ticks':int(len(ix)),'success':bool(ng[ix[-1],clone,0]>1),
               'final_angle':float(ng[ix[-1],clone,0]),'initial_angle':float(raw[ix[0],clone,26]),
               'sums':{k:float(v[ix,clone].sum()) for k,v in components.items()},
               'discounted':{str(g):{k:float(np.sum(v[ix,clone]*(g**np.arange(len(ix))))) for k,v in components.items() if k in ['original','signed','reach','grasp','net_progress','success_event']} for g in [.99,.997,.999]}}
            row['episodes'].append(e)
    row['closed_n']=len(row['episodes']);row['closed_successes']=sum(e['success'] for e in row['episodes'])
    row['summary_by_success']={}
    for success in [False,True]:
        es=[e for e in row['episodes'] if e['success']==success]
        row['summary_by_success'][str(success)]={'n':len(es),'ticks':quant([e['ticks'] for e in es]),
            'sums':{k:quant([e['sums'][k] for e in es]) for k in components},
            'discounted_signed':{str(g):quant([e['discounted'][str(g)]['signed'] for e in es]) for g in [.99,.997,.999]}}
    if '/evaluation/' in str(path):
        model_path=ROOT/f'checkpoints/phase16_r/{arm}_seed0/step50000.pt'
        s=SAC(0,P['residual']['small_bound'],'cpu',P['sac']);s.restore(torch.load(model_path,map_location='cpu',weights_only=False))
        context=np.concatenate((2/np.pi*np.arctan((raw-mean)/std),a['base'],a['chunk'].reshape(*raw.shape[:2],70),np.eye(5,dtype=np.float32)[a['cursor']]),axis=-1).astype(np.float32)
        gs=np.full(a['done'].shape,np.nan)
        rew=a['reward_original' if arm=='C' else 'reward_signed']
        lp=a['logp'].reshape(a['done'].shape);alpha=float(s.log_alpha.exp().detach())
        for e in row['episodes']:
            clone=e['clone'];ix=np.flatnonzero((a['episode'][:,clone]==e['episode'])&a['scored'][:,clone]);ret=0.;nextlp=0.
            for tick in ix[::-1]:
                ret=float(rew[tick,clone])+.99*(ret-alpha*nextlp);gs[tick,clone]=ret;nextlp=float(lp[tick,clone])
        mask=np.zeros(a['done'].shape,bool);mask[::7]=True;mask&=np.isfinite(gs)
        ti,ci=np.where(mask);ti,ci=ti[:4096],ci[:4096]
        ctx=torch.as_tensor(context[ti,ci]);base=torch.as_tensor(a['base'][ti,ci]);delta=torch.as_tensor(a['delta'][ti,ci]);act=torch.as_tensor(a['executed'][ti,ci])
        with torch.no_grad():
            q1,q2=s.q(ctx,delta,act);pred=torch.minimum(q1,q2)[:,0].numpy()
        err=pred-gs[ti,ci]
        row['observed_soft_return_check']={'n_states':len(pred),'mean_Q':float(pred.mean()),'mean_realized_soft_return':float(gs[ti,ci].mean()),'bias':float(err.mean()),'mae':float(np.mean(abs(err))),'rmse':float(np.sqrt(np.mean(err**2))), 'mean_twinQ_gap':float((q1-q2).abs().mean()),'scope':'one realized continuation per correlated state; not ground-truth expected Q or counterfactual ranking'}
        # Audit exact many-to-one clipping invariance on actual recorded contexts.
        candidate=np.argwhere(np.abs(a['base'][:,:,:6])>.999999)
        if len(candidate):
            candidate=candidate[::max(1,len(candidate)//512)][:512];tt,cc,ax=candidate.T
            x=torch.as_tensor(context[tt,cc]);bb=torch.as_tensor(a['base'][tt,cc]);dd1=torch.zeros(len(tt),6);dd2=torch.zeros_like(dd1)
            ii=torch.arange(len(tt));axes=torch.as_tensor(ax);sgn=torch.sign(bb[ii,axes]);dd1[ii,axes]=sgn*s.bound[axes]*.25;dd2[ii,axes]=sgn*s.bound[axes]*.75
            aa1=torch.cat(((bb[:,:6]+dd1).clamp(-1,1),bb[:,6:]),-1);aa2=torch.cat(((bb[:,:6]+dd2).clamp(-1,1),bb[:,6:]),-1)
            assert torch.equal(aa1,aa2)
            with torch.no_grad():
                u,v=s.q(x,dd1,aa1);w,z=s.q(x,dd2,aa2);diff=(torch.minimum(w,z)-torch.minimum(u,v))[:,0].numpy()
            row['identical_executed_action_different_proposed_delta']={'n':len(tt),'executed_action_max_difference':0.,'Q_abs_difference':quant(abs(diff)),'scope':'exact composition aliasing; observed critic input dependence, not a physical rollout'}
        # Reproduce fixed-policy logged actions with saved checkpoint on CPU.
        js=json.loads(path.with_suffix('.json').read_text());seed=P['seeds']['residual_noise_base']+js['reset_seed']
        k=np.linspace(0,len(ti)-1,min(256,len(ti)),dtype=int);tt,cc=ti[k],ci[k]
        noise=np.stack([np.random.default_rng(np.random.SeedSequence([seed,int(cl),int(a['episode'][t,cl]),int(a['age'][t,cl])])).normal(size=6).astype(np.float32) for t,cl in zip(tt,cc)])
        with torch.no_grad():
            pol=s.policy(torch.as_tensor(context[tt,cc]),torch.as_tensor(a['base'][tt,cc]),deterministic=rel.endswith('_mean'),noise=torch.as_tensor(noise))
        row['checkpoint_logged_action_CPU_reproduction']={'n':len(k),'delta_max_difference':maxabs(pol['delta'].numpy()-a['delta'][tt,cc]),'executed_max_difference':maxabs(pol['executed'].numpy()-a['executed'][tt,cc]),'checkpoint_sha256':sha(model_path),'scope':'CPU/GPU floating point comparison, not a newly relaxed production gate'}
        del s,context
    OUT['trajectories'].append(row)
    del a

for arm in ['C','D']:
    pa=R/'baseline/seed0/A_random.initial.json';pb=R/f'evaluation/{arm}_seed0/step50000/random_sampled.initial.json'
    a={(x['clone'],x['episode']):x for x in json.loads(pa.read_text())};b={(x['clone'],x['episode']):x for x in json.loads(pb.read_text())}
    keys=sorted(a.keys()&b.keys());errors=[maxabs(np.asarray(a[k]['raw'])-np.asarray(b[k]['raw'])) for k in keys];pe=[maxabs(np.asarray(a[k]['parameters'])-np.asarray(b[k]['parameters'])) for k in keys]
    OUT['initial_pairing'].append({'arm':arm,'n':len(keys),'raw_max_difference':max(errors),'raw_exact_matches':sum(e==0 for e in errors),'parameter_max_difference':max(pe),'parameter_exact_matches':sum(e==0 for e in pe)})

# Final metrics are small, already closed independent evaluations.
OUT['final_results']=[]
for arm in ['A','C','D']:
    names=['random','validation','fixed'] if arm=='A' else ['random_sampled','random_mean','validation_sampled','validation_mean','fixed_sampled','fixed_mean']
    for name in names:
        p=R/(f'baseline/seed0/A_{name}.json' if arm=='A' else f'evaluation/{arm}_seed0/step50000/{name}.json')
        d=json.loads(p.read_text());OUT['final_results'].append({'arm':arm,'scope':name,'episodes':d['episodes'],'success':d['success'],'source_sha256':sha(p)})

print(json.dumps(OUT,indent=2,allow_nan=False))
