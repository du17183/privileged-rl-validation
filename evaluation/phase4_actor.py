"""Fixture-free deployment actor and independently gated checkpoint export."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from auxiliary_learning.gt_prediction import EncodedGaussianActor
from auxiliary_learning.multitask_encoder import MultiTaskActor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stable_privileged_rl'
CHECKPOINTS=ROOT/'checkpoints'/'stable_privileged_rl'
VARIANTS=('E0','E25','E50','E100','E200','E100RB','E100R1','E100M','Q0','QGT','QV')


class RobotPolicy(nn.Module):
    """26 robot-observation inputs, 7 normalized Panda actions, no GT head."""

    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(26,256),nn.ReLU(),
                                   nn.Linear(256,256),nn.ReLU())
        self.policy_head=nn.Linear(256,14)

    def forward(self, robot_observation):
        if robot_observation.shape[-1]!=26:
            raise ValueError('Expected 26-dimensional robot observation')
        mean=self.policy_head(self.encoder(robot_observation))[...,:7]
        return mean.tanh()


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--min-heldout-success',type=float,default=.8)
    args=parser.parse_args()
    if not 0<=args.min_heldout_success<=1:
        raise ValueError('Gate must lie in [0,1]')
    target=CHECKPOINTS/'selected'/args.variant
    target.mkdir(parents=True,exist_ok=True)
    selected=[]
    for seed in range(5):
        eval_rows=read_csv(OUT/f'eval_{args.variant}_seed{seed}.csv')
        if len(eval_rows)!=51 or int(eval_rows[-1]['env_steps'])!=500000:
            raise RuntimeError(f'Incomplete training: seed {seed}')
        chosen=max(eval_rows,key=lambda r:float(r['success_rate']))
        step=int(chosen['env_steps'])
        heldout=read_csv(OUT/f'heldout_{args.variant}_seed{seed}_best.csv')[0]
        if int(heldout['selected_steps'])!=step or int(heldout['episodes'])!=64:
            raise RuntimeError(f'Unpaired heldout: seed {seed}')
        rate=float(heldout['heldout_success_rate'])
        accepted=rate>=args.min_heldout_success
        filename=f'{args.variant}_seed{seed}_step{step}_robot_policy.pt'
        row={'variant':args.variant,'seed':seed,'selected_steps':step,
             'training_validation_success':float(chosen['success_rate']),
             'independent_heldout_success':rate,'accepted':accepted,
             'actor_checkpoint':str(target/filename) if accepted else None}
        if accepted:
            source=CHECKPOINTS/f'{args.variant}_seed{seed}'/f'step_{step}.pt'
            state=torch.load(source,map_location='cpu',weights_only=False)
            weights={key:value for key,value in state['actor'].items()
                     if key.startswith('encoder.') or key.startswith('policy_head.')}
            policy=RobotPolicy()
            policy.load_state_dict(weights,strict=True)
            source_actor=(MultiTaskActor(26,7) if args.variant=='E100M'
                          else EncodedGaussianActor(26,7))
            source_actor.load_state_dict(state['actor'])
            with torch.no_grad():
                probe=torch.randn(16,26)
                reference=source_actor(probe,deterministic=True)[0]
                exported=policy(probe)
            if exported.shape!=(16,7) or not torch.allclose(reference,exported,atol=1e-7):
                raise RuntimeError('Robot-only export differs from source policy')
            torch.save({'architecture':'RobotPolicy','robot_observation_dim':26,
                        'action_dim':7,'policy':weights,'variant':args.variant,
                        'seed':seed,'selected_env_steps':step,
                        'heldout_success_rate':rate,'uses_fixture_gt_at_inference':False},
                       target/filename)
        selected.append(row)
    manifest=target/'selection_manifest.json'
    manifest.write_text(json.dumps({'criterion':'maximum fixed 32-episode training validation; '
                                              'separate 64-episode heldout gate',
                                    'min_heldout_success':args.min_heldout_success,
                                    'selected_actors':selected},indent=2),encoding='utf-8')
    print(json.dumps({'variant':args.variant,'accepted':sum(r['accepted'] for r in selected),
                      'total':5,'manifest':str(manifest)},indent=2))


if __name__=='__main__': main()
