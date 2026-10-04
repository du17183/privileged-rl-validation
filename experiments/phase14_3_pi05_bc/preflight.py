"""One-seed real pretrained forward/backward/sampler compatibility gate."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from pi05.model import build,observation
from pi05.state_adapter import StateAdapter
from pi05.action_adapter import chunk_indices,to_environment
ROOT=Path(__file__).resolve().parents[2]

def main(a):
    torch.set_num_threads(4);m=json.loads((ROOT/'datasets/phase14_2/prepared_v1/manifest.json').read_text())
    with np.load(ROOT/'datasets/phase14_2/prepared_v1/recovery_train.npz') as h:
        raw=np.concatenate((h['robot'][:a.batch],h['environment'][:a.batch]),-1);idx,mask=chunk_indices(h['trajectory_index']);actions=h['action'][idx[:a.batch]];valid=mask[:a.batch]
    adapter=StateAdapter(m['mean'],m['std']);ids,pad=adapter.encode(raw)
    start=time.monotonic();net=build(seed=0);net.train();load_s=time.monotonic()-start
    obs=observation(raw,ids,pad,'cuda:0');y=torch.zeros(a.batch,10,32,device='cuda:0');y[:,:,:7]=torch.as_tensor(actions,device='cuda:0')
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=2.5e-5,weight_decay=.01)
    timings=[]
    for step in range(5):
        torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):err=net(obs,y);loss=(err*torch.as_tensor(valid,device='cuda:0')[:,:,None]).sum()/(32*valid.sum())
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1);opt.step();torch.cuda.synchronize();timings.append(time.monotonic()-t)
    net.eval();noise=torch.randn(a.batch,10,32,device='cuda:0');torch.cuda.synchronize();t=time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):chunk=net.sample_actions('cuda:0',obs,noise=noise,num_steps=10)
    torch.cuda.synchronize();infer=time.monotonic()-t
    altered=raw.copy();altered[:,26]+=.12;altered[:,32]+=.01
    ids2,pad2=adapter.encode(altered);obs2=observation(altered,ids2,pad2,'cuda:0')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):other=net.sample_actions('cuda:0',obs2,noise=noise,num_steps=10)
    actions,clip=to_environment(chunk.float().cpu().numpy());difference=float((chunk[:,:,:7]-other[:,:,:7]).abs().mean())
    assert np.isfinite(actions).all() and difference>0 and np.any(ids!=ids2)
    d=dict(batch=a.batch,base_loaded_strict=True,load_s=load_s,parameters=sum(p.numel() for p in net.parameters()),
       trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),lora_modules=net.lora_modules,
       finite_loss=float(loss),training_step_s=timings,samples_per_second=a.batch/np.mean(timings[1:]),
       inference_s=infer,gt_same_noise_action_difference=difference,gt_token_difference=int((ids!=ids2).sum()),
       action_clip_fraction=clip,peak_gpu_GB=torch.cuda.max_memory_allocated()/1e9,torch=torch.__version__,images_used=False,rl_updates=0,padding_recipe='full32_zero_target_v2')
    out=ROOT/'results/phase14_3_pi05_bc';out.mkdir(parents=True,exist_ok=True);(out/f'preflight_b{a.batch}.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32);main(p.parse_args())
