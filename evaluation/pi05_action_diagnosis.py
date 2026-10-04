"""Approximate-neighbor action comparison; not a proof of valid multiple routes.

Uses only checkpoints already chosen by independent closed-loop validation.
The generated modes cannot be cherry-picked for the formal closed-loop tests.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from pi05.model import build,observation,restore
from pi05.state_adapter import StateAdapter
from pi05.action_adapter import to_environment
from experiments.phase14_2_recovery_bc.model import load
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/phase14_3_pi05_bc'

def benchmark_pi(net,adapter,raw):
    out={}
    for b in [1,32]:
        x=raw[:b];values=[]
        for repeat in range(13):
            torch.cuda.synchronize();start=time.monotonic()
            tokens,mask=adapter.encode(x);obs=observation(x,tokens,mask,'cuda:0')
            noise=torch.tensor(np.random.default_rng(143970+repeat).normal(size=(b,10,32)).astype(np.float32),device='cuda:0')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):y=net.sample_actions('cuda:0',obs,noise=noise,num_steps=10)
            _=to_environment(y.float().cpu().numpy());torch.cuda.synchronize()
            if repeat>=3:values.append(time.monotonic()-start)
        out[str(b)]=dict(mean_s=float(np.mean(values)),p95_s=float(np.quantile(values,.95)),warmup=3,repeats=10,
            replanning_budget_s=5/60,includes_tokenization_and_transfers=True,excludes_rpc_and_physics=True)
    return out

def benchmark_mse(policy,raw,mean,std):
    out={}
    for b in [1,32]:
        x=torch.tensor(raw[:b],device='cuda:0',dtype=torch.float32);values=[]
        for repeat in range(53):
            torch.cuda.synchronize();start=time.monotonic()
            with torch.no_grad():y=policy((x-mean)/std)
            _=y.cpu().numpy();torch.cuda.synchronize()
            if repeat>=3:values.append(time.monotonic()-start)
        out[str(b)]=dict(mean_s=float(np.mean(values)),p95_s=float(np.quantile(values,.95)),warmup=3,repeats=50,
            replanning_budget_s=1/60,includes_gpu_to_cpu=True)
    return out

def measures(pred,first,second):
    # pred [states, draws, 7]. For an MSE policy draws=1.
    v=second-first;length=np.linalg.norm(v,axis=-1);w=pred-first[:,None]
    t=(w*v[:,None]).sum(-1)/(length[:,None]**2+1e-12)
    perpendicular=np.linalg.norm(w-t[...,None]*v[:,None],axis=-1)/(length[:,None]+1e-12)
    da=np.linalg.norm(pred-first[:,None],axis=-1);db=np.linalg.norm(pred-second[:,None],axis=-1)
    mean_region=(t>=.25)&(t<=.75)&(perpendicular<=.25)
    count=(da<=db).sum(1)
    return dict(midpoint_region_fraction=float(mean_region.mean()),mean_projection_t=float(t.mean()),
        mean_perpendicular_relative=float(perpendicular.mean()),nearest_label_distance=float(np.minimum(da,db).mean()),
        query_label_mse=float(np.mean((pred-first[:,None])**2)),
        both_neighbor_modes_sampled_fraction=float(np.mean((count>=2)&(count<=pred.shape[1]-2))) if pred.shape[1]>1 else None,
        mean_generated_action_std=float(pred.std(1).mean()),draws=pred.shape[1],
        caution='The second label belongs to a nearby different state. Two generated directions do not establish two successful modes at the query state.')

def main(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    selection=json.loads((R/'selection.json').read_text());bank=dict(np.load(R/'action_pair_bank.npz'));raw=bank['raw'];n=len(raw)
    out=R/'action_diagnosis';out.mkdir(exist_ok=True);net=build();net.eval();records={}
    heldout={};rng=np.random.default_rng(143951)
    for source in ['base','recovery']:
        with np.load(ROOT/f'datasets/phase14_2/prepared_v1/{source}_test.npz') as h:
            ids=rng.choice(len(h['robot']),512,replace=False)
            heldout[source]=(np.concatenate((h['robot'][ids],h['environment'][ids]),-1),h['action'][ids])
    for seed in range(5):
        path=out/f'seed{seed}.json'
        if path.exists():continue
        result={};predictions={}
        for arm in ['A','B','C','D']:
            step=selection[f'{arm}_seed{seed}']['step'];cp=ROOT/f'checkpoints/phase14_3_pi05_bc/{arm}_seed{seed}_step{step}.pt'
            if arm in ['A','B']:
                policy,mean,std,meta=load(cp,'cuda:0')
                x=(torch.tensor(raw,device='cuda:0',dtype=torch.float32)-mean)/std
                with torch.no_grad():pred=policy(x).cpu().numpy()[:,None]
                predictions[arm]=pred;result[arm]=measures(pred,bank['expert'],bank['neighbor_expert'])
                result[arm]['inference_benchmark']=benchmark_mse(policy,raw,mean,std);del policy
            else:
                meta=restore(net,cp);adapter=StateAdapter(meta['mean'],meta['std']);tokens,mask=adapter.encode(raw);obs=observation(raw,tokens,mask,'cuda:0')
                chunks=[];latencies=[];clips=[]
                for draw in range(a.draws):
                    noise=torch.tensor(np.random.default_rng(143920+draw).normal(size=(n,10,32)).astype(np.float32),device='cuda:0')
                    torch.cuda.synchronize();tick=time.monotonic()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):chunk=net.sample_actions('cuda:0',obs,noise=noise,num_steps=10)
                    torch.cuda.synchronize();latencies.append(time.monotonic()-tick)
                    chunk,clip=to_environment(chunk.float().cpu().numpy());clips.append(clip);chunks.append(chunk)
                chunks=np.stack(chunks,1);pred=chunks[:,:,0];predictions[arm]=pred;predictions[arm+'_chunks']=chunks
                result[arm]={**measures(pred,bank['expert'],bank['neighbor_expert']),
                    'mean_chunk_generation_s':float(np.mean(latencies[1:])), 'mean_clip_fraction':float(np.mean(clips)),
                    'mean_within_chunk_action_jump_l2':float(np.linalg.norm(np.diff(chunks,axis=2),axis=-1).mean())}
                noise=torch.tensor(np.random.default_rng(143999).normal(size=(n,10,32)).astype(np.float32),device='cuda:0')
                outputs={}
                for kind in ['full','all_gt','handle_pose']:
                    altered=raw.copy()
                    if kind=='all_gt':altered[:,26:]=np.asarray(meta['mean'])[26:]
                    elif kind=='handle_pose':altered[:,32:]=np.asarray(meta['mean'])[32:]
                    tok,msk=adapter.encode(altered)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):y=net.sample_actions('cuda:0',observation(altered,tok,msk,'cuda:0'),noise=noise,num_steps=10)
                    outputs[kind]=to_environment(y.float().cpu().numpy())[0]
                result[arm]['same_noise_gt_action_sensitivity']={k:float(np.sqrt(np.mean((v-outputs['full'])**2))) for k,v in outputs.items() if k!='full'}
                result[arm]['inference_benchmark']=benchmark_pi(net,adapter,raw)
                predictions[arm+'_gt_diagnostics']=np.stack([outputs[k] for k in ['full','all_gt','handle_pose']])
            # Same held-out transition IDs for all methods/seeds. Single flow
            # sample, no best-of-N selection, first emitted control action.
            errors={}
            if arm in ['A','B']:policy,mean,std,_=load(cp,'cuda:0')
            for source,(states,labels) in heldout.items():
                ys=[]
                for k in range(0,len(states),128):
                    x=states[k:k+128]
                    if arm in ['A','B']:
                        with torch.no_grad():y=policy((torch.tensor(x,device='cuda:0',dtype=torch.float32)-mean)/std).cpu().numpy()
                    else:
                        tok,msk=adapter.encode(x)
                        noise=torch.tensor(np.random.default_rng(143952+k).normal(size=(len(x),10,32)).astype(np.float32),device='cuda:0')
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):y=net.sample_actions('cuda:0',observation(x,tok,msk,'cuda:0'),noise=noise,num_steps=10)
                        y=to_environment(y.float().cpu().numpy())[0][:,0]
                    ys.append(y)
                ys=np.concatenate(ys);predictions[arm+'_'+source+'_heldout_actions']=ys
                errors[source]=float(np.mean((ys-labels)**2))
            result[arm]['heldout_first_action_mse']=errors
            if arm in ['A','B']:del policy
        np.savez_compressed(out/f'seed{seed}.npz',**predictions,**bank);path.write_text(json.dumps(result,indent=2));records[str(seed)]=result
        print('ACTION DIAGNOSIS SEED '+str(seed),flush=True)
    d={str(s):json.loads((out/f'seed{s}.json').read_text()) for s in range(5)}
    (out/'summary.json').write_text(json.dumps(dict(seeds=d,draws=a.draws,
        interpretation='Approximate-neighbor evidence only. Combine with all four closed-loop arms; no causal claim that MSE averaging explains failures.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draws',type=int,default=16);main(p.parse_args())
