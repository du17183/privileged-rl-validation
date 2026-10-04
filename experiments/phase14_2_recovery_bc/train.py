"""Seven independent BC fits vectorized across arm, identical Adam/MSE recipe."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from experiments.phase13_random_expert_bc.model import Policy
from experiments.phase13_random_expert_bc.bc import mse
from .model import inputs,FEATURES
ARMS={'A':('robot',None),'B':('full',None),'C':('robot','recovery'),'D':('full','recovery'),
      'E':('robot','ordinary'),'F':('full','ordinary'),'D_no_handle':('no_handle','recovery')}

class Ensemble(nn.Module):
    def __init__(self,initial):
        super().__init__();self.names=list(initial.state_dict())
        self.weights=nn.ParameterList([nn.Parameter(v.detach().clone().expand(len(ARMS),*v.shape).clone())
                                      for k,v in initial.state_dict().items() if k!='log_std'])
    def forward(self,x):
        w,b,w2,b2,w3,b3=self.weights
        x=torch.relu(torch.bmm(x,w.transpose(1,2))+b[:,None])
        x=torch.relu(torch.bmm(x,w2.transpose(1,2))+b2[:,None])
        return torch.tanh(torch.bmm(x,w3.transpose(1,2))+b3[:,None])
    def extract(self,i,initial):
        state={k:v.detach().clone() for k,v in initial.state_dict().items()}
        for k,v in zip([k for k in self.names if k!='log_std'],self.weights):state[k]=v[i].detach().clone()
        return state

def main(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/f'bc_seed{a.seed}.csv').exists():raise FileExistsError(root)
    m=json.loads((Path(a.data)/'manifest.json').read_text());device=torch.device(a.device)
    mean=torch.tensor(m['mean'],device=device);std=torch.tensor(m['std'],device=device);data={}
    for s in ['base','recovery','ordinary']:
        for k in ['train','validation','test']:
            d=dict(np.load(Path(a.data)/f'{s}_{k}.npz'));r=torch.tensor(d['robot'],device=device);e=torch.tensor(d['environment'],device=device)
            data[s,k]={mode:inputs(r,e,mean,std,mode) for mode in ['robot','full','no_handle']}
            data[s,k].update(action=torch.tensor(d['action'],device=device),index=d['trajectory_index'])
    initial=Policy().to(device);ensemble=Ensemble(initial).to(device);opt=torch.optim.Adam(ensemble.parameters(),lr=3e-4)
    # Verify batched forward is exactly the original model before training.
    probe=data['base','train']['full'][:32]
    if not torch.allclose(ensemble(probe.expand(len(ARMS),-1,-1))[0],initial(probe),atol=1e-6):raise RuntimeError('ensemble forward mismatch')
    torch.save(dict(model=initial.state_dict(),seed=a.seed),root/f'initial_seed{a.seed}.pt')
    gen=torch.Generator(device=device).manual_seed(142400+a.seed);best={v:float('inf') for v in ARMS};started=time.monotonic()
    with (root/f'bc_seed{a.seed}.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['update','variant','train_mse','validation_mse','elapsed_s']);writer.writeheader()
        for step in range(1,a.updates+1):
            idx={s:torch.randint(len(data[s,'train']['action']),(512 if s=='base' else 256,),device=device,generator=gen) for s in ['base','recovery','ordinary']}
            xs=[];ys=[]
            for arm,(mode,source) in ARMS.items():
                d=data['base','train'];x=d[mode][idx['base']];y=d['action'][idx['base']]
                if source:
                    t=data[source,'train'];x=torch.cat((x[:256],t[mode][idx[source]]));y=torch.cat((y[:256],t['action'][idx[source]]))
                xs.append(x);ys.append(y)
            opt.zero_grad(set_to_none=True);loss=(ensemble(torch.stack(xs))-torch.stack(ys)).square().mean((1,2));loss.sum().backward();opt.step()
            if step%1000==0 or step==a.updates:
                for i,(arm,(mode,source)) in enumerate(ARMS.items()):
                    net=Policy().to(device);net.load_state_dict(ensemble.extract(i,initial))
                    scores={s:mse(net,data[s,'validation'][mode],data[s,'validation']['action'])['transition_mse'] for s in ['base','recovery','ordinary']};score=float(np.mean(list(scores.values())))
                    row=dict(update=step,variant=arm,train_mse=float(loss[i].detach()),validation_mse=score,elapsed_s=time.monotonic()-started)
                    writer.writerow(row);print(json.dumps(row),flush=True)
                    payload=dict(model=net.state_dict(),mean=m['mean'],std=m['std'],mode=mode,variant=arm,seed=a.seed,update=step,
                                 validation_mse=score,validation_by_source=scores,dataset_sha256=m['dataset_sha256'],
                                 parameter_count=sum(p.numel() for p in initial.parameters()),input_dim=39,target_angle=1.,environment_features=FEATURES)
                    if score<best[arm]:best[arm]=score;torch.save(payload,root/f'{arm}_seed{a.seed}_best.pt')
                    if step==a.updates:torch.save(payload,root/f'{arm}_seed{a.seed}_final.pt')
                f.flush()
    results=[]
    for arm,(mode,source) in ARMS.items():
        meta=torch.load(root/f'{arm}_seed{a.seed}_best.pt',map_location=device,weights_only=False);net=Policy().to(device);net.load_state_dict(meta['model'])
        for s in ['base','recovery','ordinary']:
            d=data[s,'test'];results.append(dict(variant=arm,seed=a.seed,source=s,selected_update=meta['update'],**mse(net,d[mode],d['action'],d['index'])))
    (root/f'bc_seed{a.seed}_summary.json').write_text(json.dumps(dict(test=results,updates=a.updates,paired_initialization=True,paired_sampling=True,
          batch=512,examples_per_arm=a.updates*512,extra_source_examples=a.updates*256,elapsed_s=time.monotonic()-started),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--updates',type=int,default=20000);p.add_argument('--device',default='cuda:0');main(p.parse_args())
