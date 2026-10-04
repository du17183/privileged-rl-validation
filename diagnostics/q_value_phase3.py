"""Best/final Q, TD and expert-direction probes for all Phase 3 variants."""

import csv
from pathlib import Path

import torch

from algorithms.asymmetric_sac import AsymmetricSAC
from algorithms.door_phase3_sac import MaskedSAC
from auxiliary_learning.gt_prediction import AuxiliarySAC
from diagnostics.q_value_analysis import fixed_probe, checkpoint_probe
from privileged_variants import gt_minimal, gt_medium, gt_full


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'door_privileged_ablation'
VARIANTS = ('diag_A','diag_B','B1','B2','B5','B6','E0','E1','E2','E3')


def model(variant, switched=False):
    if variant == 'diag_A' or (variant == 'B6' and switched):
        return AsymmetricSAC(26,11,7,'A',device='cpu')
    if variant in ('E0','E1','E2','E3'):
        return AuxiliarySAC(26,11,7,device='cpu',aux_weight=0)
    indices = gt_minimal.INDICES if variant=='B1' else (
        gt_medium.INDICES if variant=='B2' else gt_full.INDICES)
    return MaskedSAC(26,11,7,indices,device='cpu')


def selected_step(variant,seed,mode):
    with (OUT/f'eval_{variant}_seed{seed}.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    return int((max(rows,key=lambda r:float(r['success_rate'])) if mode=='best' else rows[-1])['env_steps'])


def main():
    torch.set_num_threads(4)
    batch=fixed_probe(ROOT/'door_dataset'/'door_expert_1000.h5')
    path=OUT/'fixed_probe_phase3_best_final.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=None
        for variant in VARIANTS:
            for seed in range(5):
                for mode in ('best','final'):
                    step=selected_step(variant,seed,mode)
                    checkpoint=ROOT/'checkpoints'/'door_privileged_ablation'/f'{variant}_seed{seed}'/f'step_{step}.pt'
                    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    agent=model(variant, bool(state.get('switched', False)))
                    agent.actor.load_state_dict(state['actor'])
                    agent.critic.load_state_dict(state['critic'])
                    agent.target_critic.load_state_dict(state['target_critic'])
                    agent.log_alpha.data.copy_(state['log_alpha'])
                    row={'variant':variant,'seed':seed,'mode':mode,'env_steps':step,
                         **checkpoint_probe(agent,batch)}
                    if writer is None:
                        writer=csv.DictWriter(stream,fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
            print('probed',variant,flush=True)
    print('Wrote',path)


if __name__=='__main__':
    main()
