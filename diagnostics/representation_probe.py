"""Linear probe on frozen E0/E1 actor encoders and common heldout states."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from auxiliary_learning.gt_prediction import EncodedGaussianActor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'door_privileged_ablation'


def linear_probe(x, y, train, test, ridge=10.0):
    mu = x[train].mean(axis=0)
    scale = x[train].std(axis=0) + 1e-5
    x = (x - mu) / scale
    x = np.concatenate((x, np.ones((len(x),1))),axis=1)
    xt = x[train]
    reg = np.eye(x.shape[1]) * ridge
    reg[-1,-1] = 0
    beta = np.linalg.solve(xt.T @ xt + reg, xt.T @ y[train])
    return x[test] @ beta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',nargs='+',type=int,default=[0,500000])
    parser.add_argument('--output',type=Path,default=OUT/'representation_probe.csv')
    args=parser.parse_args()
    torch.set_num_threads(4)
    probe = np.load(OUT / 'representation_probe_states.npz')
    robot, gt, env_id = probe['robot'], probe['gt'], probe['env_id']
    train, test = env_id < 16, env_id >= 16
    if min(train.sum(), test.sum()) < 100:
        raise RuntimeError('Insufficient independent environments for linear probe')
    rows = []
    for variant in ('E0','E1','E2','E3'):
        for seed in range(5):
            for step in args.steps:
                actor = EncodedGaussianActor(26,7)
                path = ROOT / 'checkpoints/door_privileged_ablation' / f'{variant}_seed{seed}' / f'step_{step}.pt'
                actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['actor'])
                actor.eval()
                with torch.no_grad():
                    latent = actor.encoder(torch.from_numpy(robot).float()).numpy().astype(np.float64)
                angle = gt[:,0].astype(np.float64)
                angle_hat = linear_probe(latent, angle, train, test)
                mae = float(np.mean(np.abs(angle_hat - angle[test])))
                baseline = float(np.mean(np.abs(angle[train].mean() - angle[test])))
                r2 = float(1 - np.sum((angle_hat-angle[test])**2) /
                           np.sum((angle[test]-angle[test].mean())**2))
                contact_brier = []
                prevalence = []
                for j in (9,10):
                    label = gt[:,j].astype(np.float64)
                    pred = np.clip(linear_probe(latent,label,train,test),0,1)
                    contact_brier.append(float(np.mean((pred-label[test])**2)))
                    prevalence.append(float(label[test].mean()))
                rows.append({'variant':variant,'seed':seed,'env_steps':step,
                             'heldout_states':int(test.sum()),
                             'angle_probe_mae_rad':mae,'constant_angle_mae_rad':baseline,
                             'angle_probe_r2':r2,'contact_probe_brier':float(np.mean(contact_brier)),
                             'contact_prevalence':float(np.mean(prevalence))})
    with args.output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f'Wrote {args.output}')


if __name__=='__main__':
    main()
