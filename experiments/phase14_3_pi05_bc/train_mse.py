"""Original Policy, original MSE/Adam, keep all closed-loop selection candidates."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from experiments.phase13_random_expert_bc.model import Policy
from experiments.phase14_2_recovery_bc.model import inputs,FEATURES
ROOT=Path(__file__).resolve().parents[2]

def main(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
    directory=ROOT/'checkpoints/phase14_3_pi05_bc';directory.mkdir(parents=True,exist_ok=True)
    early=getattr(a,'early_only',False)
    complete=directory/(f'mse_seed{a.seed}_'+('early_complete.json' if early else 'complete.json'))
    if complete.exists():return
    data=ROOT/'datasets/phase14_2/prepared_v1';m=json.loads((data/'manifest.json').read_text());device='cuda:0'
    mean=torch.tensor(m['mean'],device=device);std=torch.tensor(m['std'],device=device);ds={}
    for source in ['base','recovery']:
        with np.load(data/f'{source}_train.npz') as d:
            ds[source]=(inputs(torch.tensor(d['robot'],device=device),torch.tensor(d['environment'],device=device),mean,std,'full'),torch.tensor(d['action'],device=device))
    initial=Policy().to(device);init={k:v.detach().clone() for k,v in initial.state_dict().items()}
    costs={}
    for arm in ['A','B']:
        net=Policy().to(device);net.load_state_dict(init);opt=torch.optim.Adam(net.parameters(),lr=3e-4)
        gen=torch.Generator(device=device).manual_seed(143400+a.seed);start=time.monotonic()
        file=directory/f'{"early_" if early else ""}{arm}_seed{a.seed}_training.csv'
        with file.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=['update','train_mse','elapsed_s']);w.writeheader()
            for step in range(1,1001 if early else 20001):
                si=torch.randint(len(ds['base'][0]),(512,),device=device,generator=gen)
                ri=torch.randint(len(ds['recovery'][0]),(256,),device=device,generator=gen)
                x,y=ds['base'][0][si],ds['base'][1][si]
                if arm=='B':x=torch.cat((x[:256],ds['recovery'][0][ri]));y=torch.cat((y[:256],ds['recovery'][1][ri]))
                opt.zero_grad(set_to_none=True);loss=(net(x)-y).square().mean();loss.backward();opt.step()
                if step%1000==0:w.writerow(dict(update=step,train_mse=float(loss.detach()),elapsed_s=time.monotonic()-start));stream.flush()
                if step in [1000,5000,10000,15000,20000]:
                    meta=dict(model=net.state_dict(),mean=m['mean'],std=m['std'],mode='full',variant=arm,seed=a.seed,update=step,
                           parameter_count=77838,dataset_sha256=m['dataset_sha256'],input_dim=39,target_angle=1,environment_features=FEATURES)
                    torch.save(meta,directory/f'{arm}_seed{a.seed}_step{step}.pt')
        costs[arm]=dict(updates=1000 if early else 20000,batch=512,elapsed_s=time.monotonic()-start,examples=512000 if early else 10240000,
           early_checkpoint_reconstruction=early,main_budget_unchanged=20000)
    complete.write_text(json.dumps(costs,indent=2));print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--early-only',action='store_true');main(p.parse_args())
