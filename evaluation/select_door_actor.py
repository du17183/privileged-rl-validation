"""Export robot-only actors selected with fixture-measured Door success.

Selection uses the training validation curve. A separate heldout fixture
evaluation gates export. Exported payload contains no critic or GT input.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'door_privileged_ablation'
CHECKPOINTS = ROOT / 'checkpoints' / 'door_privileged_ablation'


def one_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('E0','E1','E2','E3'),default='E2')
    parser.add_argument('--min-heldout-success',type=float,default=0.8)
    args=parser.parse_args()
    if not 0 <= args.min_heldout_success <= 1:
        raise ValueError('min-heldout-success must be in [0,1]')
    target=CHECKPOINTS/'selected'/args.variant
    target.mkdir(parents=True,exist_ok=True)
    report=[]
    for seed in range(5):
        eval_rows=one_csv(RESULTS/f'eval_{args.variant}_seed{seed}.csv')
        if len(eval_rows)!=21 or int(eval_rows[-1]['env_steps'])!=500000:
            raise RuntimeError(f'Incomplete training evaluation for seed {seed}')
        chosen=max(eval_rows,key=lambda r:float(r['success_rate']))
        step=int(chosen['env_steps'])
        heldout=one_csv(RESULTS/f'heldout_{args.variant}_seed{seed}_best.csv')[0]
        if int(heldout['episodes'])!=64 or int(heldout['selected_steps'])!=step:
            raise RuntimeError(f'Unpaired heldout checkpoint: seed {seed}')
        heldout_success=float(heldout['heldout_success_rate'])
        accepted=heldout_success >= args.min_heldout_success
        name=f'{args.variant}_seed{seed}_step{step}_actor_only.pt'
        row={'variant':args.variant,'seed':seed,'selected_steps':step,
             'selection_success_rate':float(chosen['success_rate']),
             'independent_heldout_success_rate':heldout_success,
             'heldout_episodes':64,'accepted':accepted,
             'actor_checkpoint':str(target/name) if accepted else None}
        if accepted:
            source=CHECKPOINTS/f'{args.variant}_seed{seed}'/f'step_{step}.pt'
            state=torch.load(source,map_location='cpu',weights_only=False)
            if 'actor' not in state or 'variant_label' not in state:
                raise RuntimeError(f'Malformed source checkpoint: {source}')
            if state['variant_label']!=args.variant:
                raise RuntimeError(f'Variant mismatch in {source}')
            torch.save({'architecture':'EncodedGaussianActor','robot_observation_dim':26,
                        'action_dim':7,'actor':state['actor'],
                        'variant':args.variant,'seed':seed,'selected_env_steps':step,
                        'heldout_success_rate':heldout_success,
                        'uses_fixture_GT_at_inference':False},target/name)
        report.append(row)
    manifest=target/'selection_manifest.json'
    manifest.write_text(json.dumps({'min_heldout_success':args.min_heldout_success,
                                    'criterion':'max 64-episode training validation success; separate 64-episode heldout gate',
                                    'selected_actors':report},indent=2),encoding='utf-8')
    print(json.dumps({'accepted':sum(r['accepted'] for r in report),
                      'total':len(report),'manifest':str(manifest)},indent=2))


if __name__=='__main__':
    main()
