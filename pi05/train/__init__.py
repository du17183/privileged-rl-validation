"""Fixed-budget pretrained pi05 LoRA flow-matching imitation training."""
import argparse,csv,json,math,time,os
from pathlib import Path
import numpy as np,torch
from pi05.model import build,observation,learned_state,BASE
from pi05.data_adapter import DATA,ROOT

def main(a):
    assert a.arm in ['C','D'];torch.set_num_threads(4);torch.manual_seed(a.seed)
    output=ROOT/'checkpoints/phase14_3_pi05_bc';output.mkdir(parents=True,exist_ok=True)
    if a.updates!=5000:output=output/'pilot';output.mkdir(exist_ok=True)
    complete=output/f'{a.arm}_seed{a.seed}_complete.json'
    if complete.exists():return
    m=json.loads((DATA/'manifest.json').read_text());ds={}
    for source in ['base','recovery']:
        with np.load(DATA/f'{source}_train.npz') as h:
            ds[source]={k:torch.tensor(h[k],device='cuda:0') for k in ['raw','tokens','token_mask','actions','action_mask']}
    start=time.monotonic();net=build(seed=a.seed);net.train();load_s=time.monotonic()-start
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01)
    gen=torch.Generator(device='cuda:0').manual_seed(143400+a.seed);torch.manual_seed(143450+a.seed)
    saved=[];times=[]
    with (output/f'{a.arm}_seed{a.seed}_training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['update','flow_loss','gradient_norm','lr','elapsed_s','step_s']);w.writeheader()
        for step in range(1,a.updates+1):
            torch.cuda.synchronize();tick=time.monotonic()
            b=a.batch;si=torch.randint(len(ds['base']['raw']),(b,),device='cuda:0',generator=gen)
            ri=torch.randint(len(ds['recovery']['raw']),(b//2,),device='cuda:0',generator=gen)
            batch={k:v[si] for k,v in ds['base'].items()}
            if a.arm=='D':batch={k:torch.cat((batch[k][:b//2],ds['recovery'][k][ri])) for k in batch}
            obs=observation(batch['raw'],batch['tokens'],batch['token_mask'],'cuda:0')
            target=torch.zeros(b,10,32,device='cuda:0');target[:,:,:7]=batch['actions']
            lr=1e-4*min(1,step/250)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-250)/max(1,a.updates-250))))
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                # All32 dimensions enter the action embedding and official
                # Euler sampler. Known-zero padding must learn its flow too.
                # Only illegal horizon steps are masked, never padding motors.
                error=net(obs,target);loss=(error*batch['action_mask'][:,:,None]).sum()/(32*batch['action_mask'].sum())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite flow loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1);opt.step();torch.cuda.synchronize();dt=time.monotonic()-tick;times.append(dt)
            if step==1 or step%20==0:
                row=dict(update=step,flow_loss=float(loss.detach()),gradient_norm=float(norm),lr=lr,elapsed_s=time.monotonic()-start,step_s=dt)
                w.writerow(row);f.flush();print(json.dumps(row),flush=True)
            if step in [250,1250,2500,3750,5000] or step==a.updates:
                payload=dict(learned=learned_state(net),variant=a.arm,seed=a.seed,update=step,mode='full',input_dim=39,
                    data_manifest=m['original_manifest'],mean=m['mean'],std=m['std'],model='official_openpi_pi05_state_only_lora16',
                    images_used=False,chunk_horizon=10,execute_steps=5,flow_steps=10,trainable=sum(p.numel() for p in params),
                    parameters=sum(p.numel() for p in net.parameters()),base_checkpoint=str(BASE),
                    target_angle=1.,normalization='Phase14.2 train-S z stats then intrinsic atan/discrete pi05 state adaptor',
                    batch=b,lr_peak=1e-4,selected_by_test=False,rl_updates=0,padding_recipe='full32_zero_target_v2')
                dest=output/f'{a.arm}_seed{a.seed}_step{step}.pt';tmp=dest.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,dest)
                dest.with_suffix('.ready.json').write_text(json.dumps({'checkpoint':str(dest),'update':step,'complete':True}))
                saved.append(str(dest))
    complete.write_text(json.dumps(dict(seed=a.seed,arm=a.arm,updates=a.updates,batch=a.batch,load_s=load_s,
        elapsed_s=time.monotonic()-start,mean_step_s=float(np.mean(times[20:])),anchor_draws=a.updates*a.batch,
        peak_gpu_GB=torch.cuda.max_memory_allocated()/1e9,checkpoints=saved,rl_updates=0),indent=2));print('TRAIN COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--updates',type=int,default=5000);p.add_argument('--batch',type=int,default=128);main(p.parse_args())
