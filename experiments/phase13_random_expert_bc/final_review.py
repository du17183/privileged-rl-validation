"""Check experiment identity, pair matching, budgets and preserved baselines."""
import csv
import hashlib
import json
from pathlib import Path
import torch
from experiments.phase13_random_expert_bc.inventory import snapshot

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'


def parameters(path):
    rows=list(csv.DictReader(Path(path).open()))
    columns=['clone','episode','angle0_rad','offset_x','offset_y','offset_z','friction_scale']
    return sorted(tuple(row[c] for c in columns) for row in rows)


def main():
    issues=[];pairs=[]
    for seed in range(5):
        a=parameters(R/f'heldout/A_seed{seed}_best.csv');b=parameters(R/f'heldout/B_seed{seed}_best.csv')
        same=a==b and len(a)==128
        pairs.append(dict(seed=seed,episodes=len(a),reset_parameters_exact_match=same))
        if not same:issues.append(f'BC reset pair seed{seed}')
    heldout_pairs=[]
    for seed in range(3):
        reference=parameters(R/f'rl_heldout/A_seed{seed}_final_deterministic.csv')
        same=all(parameters(R/f'rl_heldout/{arm}_seed{seed}_final_deterministic.csv')==reference for arm in ['B','C'])
        heldout_pairs.append(dict(seed=seed,episodes=len(reference),reset_parameters_exact_match=same))
        if not same or len(reference)!=128:issues.append(f'RL heldout reset pair {seed}')
        for arm in ['A','B','C']:
            test=json.loads((R/f'rl_heldout/{arm}_seed{seed}_final_deterministic.json').read_text())
            if test['episodes']!=128:issues.append(f'RL heldout count {arm}/{seed}')
        stochastic=json.loads((R/f'rl_heldout/C_seed{seed}_final_stochastic.json').read_text())
        if stochastic['episodes']!=64:issues.append(f'RL stochastic count {seed}')
    chosen=json.loads((R/'anchor_validation.json').read_text())
    source=torch.load(ROOT/chosen['source'],map_location='cpu',weights_only=False)
    rl=[]
    for seed in range(3):
        for arm in ['A','B','C']:
            path=ROOT/f'checkpoints/phase13_random_expert_bc/rl/{arm}_seed{seed}'
            initial=torch.load(path/'step_0.pt',map_location='cpu',weights_only=False)
            same=all(torch.equal(value,initial['model'][key]) for key,value in source['model'].items())
            summary=json.loads((R/f'rl/{arm}_seed{seed}/summary.json').read_text())
            record=dict(arm=arm,seed=seed,initial_model_exact_match=same,interactions=summary['interactions'],
                        updates=summary['updates'],std_cap=summary['controls']['std'],rollbacks=summary['controls']['rollbacks'])
            if not same or summary['interactions']!=50016 or summary['controls']['std']!=.01:issues.append(f'RL identity/budget {arm}/{seed}')
            if arm=='A':
                final=torch.load(path/'final.pt',map_location='cpu',weights_only=False)
                frozen=all(torch.equal(value,final['model'][key]) for key,value in source['model'].items())
                record['frozen_actor_exact_match']=frozen
                if not frozen or summary['updates']!=0:issues.append(f'Frozen baseline changed {seed}')
            elif summary['updates']!=6000:issues.append(f'Update ratio {arm}/{seed}')
            rl.append(record)
    prior=json.loads((R/'baseline_inventory.json').read_text());now=snapshot()
    changes={kind:[name for name,value in prior[kind].items() if now[kind].get(name)!=value] for kind in ['sources','artifacts']}
    if any(changes.values()):issues.append('Prior artifacts changed')
    dataset=ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5'
    split=json.loads((ROOT/'datasets/random_door_expert/split_v1/split.json').read_text())
    same_hash=hashlib.sha256(dataset.read_bytes()).hexdigest()==split['dataset_sha256']
    if not same_hash:issues.append('Dataset changed')
    result=dict(issues=issues,passed=not issues,bc_reset_pairs=pairs,rl_heldout_reset_pairs=heldout_pairs,rl_identity=rl,
                dataset_hash_unchanged=same_hash,prior_preservation=changes,
                old_sources=len(prior['sources']),old_artifacts=len(prior['artifacts']),
                training_source={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    (ROOT/'experiments/phase13_random_expert_bc').glob('*.py')},
                clarification='Source hashes describe final reproducible implementation. Earlier archived phase13_source_* and phase13_rl_v1.tar retain staged development versions. No trained algorithm source was changed after RL launch.')
    (R/'final_review.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['training_source','rl_identity']},indent=2))
    if issues:raise RuntimeError(issues)


if __name__=='__main__':main()
