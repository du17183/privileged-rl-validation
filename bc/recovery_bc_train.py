"""Six matched arms: data coverage x environment information."""
import argparse,copy,csv,json,time
from pathlib import Path
import numpy as np,torch
from experiments.phase13_random_expert_bc.model import Policy,inputs
from experiments.phase13_random_expert_bc.bc import mse
from experiments.phase14_recovery_bc.gate import require

ARMS={'A':('A',None),'B':('B',None),'C':('A','recovery'),'D':('B','recovery'),'E':('A','ordinary'),'F':('B','ordinary')}


def main(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/f'bc_seed{a.seed}.csv').exists():raise FileExistsError(root)
    manifest=json.loads((Path(a.data)/'manifest.json').read_text());device=torch.device(a.device)
    verification=manifest['recovery_validation']
    require(verification)
    mean=torch.tensor(manifest['mean'],device=device);std=torch.tensor(manifest['std'],device=device)
    data={}
    for source in ['base','recovery','ordinary']:
        for split in ['train','validation','test']:
            arrays=dict(np.load(Path(a.data)/f'{source}_{split}.npz'))
            robot=torch.tensor(arrays['robot'],device=device);env=torch.tensor(arrays['environment'],device=device)
            data[source,split]=dict(A=inputs(robot,env,mean,std,'A'),B=inputs(robot,env,mean,std,'B'),
                action=torch.tensor(arrays['action'],device=device),index=arrays['trajectory_index'])
    initial=Policy().to(device);policies={arm:copy.deepcopy(initial) for arm in ARMS}
    optimizers={arm:torch.optim.Adam(p.parameters(),lr=3e-4) for arm,p in policies.items()}
    initial_hash={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
    generator=torch.Generator(device=device).manual_seed(14400+a.seed)
    best={arm:float('inf') for arm in ARMS};started=time.monotonic()
    with (root/f'bc_seed{a.seed}.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['update','variant','train_mse','validation_mse','elapsed_s']);writer.writeheader()
        for step in range(1,a.updates+1):
            baseids=torch.randint(len(data['base','train']['action']),(512,),device=device,generator=generator)
            newids=torch.randint(manifest['label_matching'],(256,),device=device,generator=generator)
            losses={}
            for arm,(mode,source) in ARMS.items():
                d=data['base','train'];x=d[mode][baseids];y=d['action'][baseids]
                if source:
                    e=data[source,'train'];x=torch.cat((x[:256],e[mode][newids]));y=torch.cat((y[:256],e['action'][newids]))
                opt=optimizers[arm];opt.zero_grad(set_to_none=True);loss=(policies[arm](x)-y).square().mean()
                loss.backward();opt.step();losses[arm]=float(loss.detach())
            if step%1000==0 or step==a.updates:
                for arm,(mode,source) in ARMS.items():
                    scores={s:mse(policies[arm],data[s,'validation'][mode],data[s,'validation']['action'])['transition_mse'] for s in ['base','recovery','ordinary']}
                    score=float(np.mean(list(scores.values())))
                    row=dict(update=step,variant=arm,train_mse=losses[arm],validation_mse=score,elapsed_s=time.monotonic()-started)
                    writer.writerow(row);print(json.dumps(row),flush=True)
                    payload=dict(model=policies[arm].state_dict(),mean=manifest['mean'],std=manifest['std'],arm=mode,variant=arm,
                        seed=a.seed,update=step,validation_mse=score,validation_by_source=scores,dataset_sha256=manifest['dataset_sha256'],
                        parameter_count=sum(p.numel() for p in initial.parameters()),input_dim=39)
                    if score<best[arm]:best[arm]=score;torch.save(payload,root/f'{arm}_seed{a.seed}_best.pt')
                    if step==a.updates:torch.save(payload,root/f'{arm}_seed{a.seed}_final.pt')
                f.flush()
    rows=[]
    for arm,(mode,source) in ARMS.items():
        payload=torch.load(root/f'{arm}_seed{a.seed}_best.pt',map_location=device,weights_only=False);policies[arm].load_state_dict(payload['model'])
        for s in ['base','recovery','ordinary']:
            d=data[s,'test'];rows.append(dict(variant=arm,seed=a.seed,source=s,selected_update=payload['update'],**mse(policies[arm],d[mode],d['action'],d['index'])))
    torch.save(dict(model=initial_hash,seed=a.seed),root/f'initial_seed{a.seed}.pt')
    (root/f'bc_seed{a.seed}_summary.json').write_text(json.dumps(dict(test=rows,updates=a.updates,paired_initialization=True,
        paired_sampling=True,batch=512,elapsed_s=time.monotonic()-started),indent=2))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=20000);p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
