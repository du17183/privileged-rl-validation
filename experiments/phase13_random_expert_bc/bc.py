"""Paired BC fits: same weights, sampled minibatches, optimizer and data."""
import argparse
import copy
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.phase13_random_expert_bc.model import Policy,inputs


@torch.no_grad()
def mse(policy,x,y,index=None):
    errors=[]
    for lo in range(0,len(x),4096):errors.append((policy(x[lo:lo+4096])-y[lo:lo+4096]).square().mean(-1).cpu())
    errors=torch.cat(errors).numpy()
    result=dict(transition_mse=float(errors.mean()))
    if index is not None:
        result['trajectory_mse']=float(np.mean([errors[index==i].mean() for i in np.unique(index)]))
    return result


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed)
    manifest=json.loads((Path(args.data)/'split.json').read_text())
    device=torch.device(args.device)
    mean=torch.tensor(manifest['mean'],device=device);std=torch.tensor(manifest['std'],device=device)
    data={}
    for name in ['train','validation','test']:
        arrays=dict(np.load(Path(args.data)/(name+'.npz')))
        robot=torch.tensor(arrays['robot'],device=device);context=torch.tensor(arrays['environment'],device=device)
        data[name]=dict(A=inputs(robot,context,mean,std,'A'),B=inputs(robot,context,mean,std,'B'),
                        action=torch.tensor(arrays['action'],device=device),index=arrays['trajectory_index'])
    initial=Policy().to(device)
    policies={arm:copy.deepcopy(initial) for arm in ['A','B']}
    optimizers={arm:torch.optim.Adam(policies[arm].parameters(),lr=3e-4) for arm in policies}
    start_weights={arm:{k:v.detach().cpu().clone() for k,v in policies[arm].state_dict().items()} for arm in policies}
    assert all(torch.equal(start_weights['A'][k],start_weights['B'][k]) for k in start_weights['A'])
    generator=torch.Generator(device=device).manual_seed(13400+args.seed)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    log=root/f'bc_seed{args.seed}.csv'
    if log.exists():raise FileExistsError(log)
    best={arm:float('inf') for arm in policies};started=time.monotonic()
    rows=[]
    with log.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['update','arm','train_mse','validation_mse','elapsed_s']);writer.writeheader()
        for step in range(1,args.updates+1):
            indices=torch.randint(len(data['train']['action']),(512,),device=device,generator=generator)
            losses={}
            for arm,policy in policies.items():
                optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                loss=(policy(data['train'][arm][indices])-data['train']['action'][indices]).square().mean()
                loss.backward();optimizer.step();losses[arm]=float(loss.detach())
            if step%1000==0 or step==args.updates:
                for arm,policy in policies.items():
                    score=mse(policy,data['validation'][arm],data['validation']['action'])['transition_mse']
                    row=dict(update=step,arm=arm,train_mse=losses[arm],validation_mse=score,elapsed_s=time.monotonic()-started)
                    writer.writerow(row);print(json.dumps(row),flush=True)
                    payload=dict(model=policy.state_dict(),mean=manifest['mean'],std=manifest['std'],arm=arm,
                                 seed=args.seed,update=step,validation_mse=score,dataset_sha256=manifest['dataset_sha256'],
                                 parameter_count=sum(p.numel() for p in policy.parameters()),input_dim=39)
                    if score<best[arm]:
                        best[arm]=score;torch.save(payload,root/f'{arm}_seed{args.seed}_best.pt')
                    if step==args.updates:torch.save(payload,root/f'{arm}_seed{args.seed}_final.pt')
                f.flush()
        for arm in policies:
            value=torch.load(root/f'{arm}_seed{args.seed}_best.pt',map_location=device,weights_only=False)
            policies[arm].load_state_dict(value['model'])
            result=mse(policies[arm],data['test'][arm],data['test']['action'],data['test']['index'])
            rows.append(dict(arm=arm,seed=args.seed,selected_update=value['update'],**result))
    (root/f'bc_seed{args.seed}_summary.json').write_text(json.dumps(dict(test=rows,paired_initialization=True,
         paired_minibatch_indices=True,updates=args.updates,parameter_count=sum(p.numel() for p in initial.parameters()),
         elapsed_s=time.monotonic()-started),indent=2))
    print('COMPLETE '+json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())
