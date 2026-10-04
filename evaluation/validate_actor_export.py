"""Check robot-only deployment exports against selected source checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from auxiliary_learning.gt_prediction import EncodedGaussianActor

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='E2')
    args=parser.parse_args()
    manifest=ROOT/'checkpoints'/'door_privileged_ablation'/'selected'/args.variant/'selection_manifest.json'
    entries=json.loads(manifest.read_text(encoding='utf-8'))['selected_actors']
    torch.manual_seed(20260929)
    robot=torch.randn(8,26)
    validated=[]
    for row in entries:
        if not row['accepted']: continue
        deployment=torch.load(row['actor_checkpoint'],map_location='cpu',weights_only=False)
        source_path=ROOT/'checkpoints'/'door_privileged_ablation'/f"{args.variant}_seed{row['seed']}"/f"step_{row['selected_steps']}.pt"
        source=torch.load(source_path,map_location='cpu',weights_only=False)
        if deployment['uses_fixture_GT_at_inference'] is not False:
            raise AssertionError('Deployment payload claims a GT input')
        if set(deployment['actor'])!=set(source['actor']) or not all(
                torch.equal(deployment['actor'][key],source['actor'][key])
                for key in source['actor']):
            raise AssertionError(f"Export weights mismatch for seed {row['seed']}")
        actor=EncodedGaussianActor(26,7)
        actor.load_state_dict(deployment['actor'])
        action,_=actor(robot,deterministic=True)
        if action.shape!=(8,7): raise AssertionError('Unexpected deploy action shape')
        validated.append(row['seed'])
    if validated!=list(range(5)):
        raise AssertionError(f'Expected five accepted seed exports, got {validated}')
    print(json.dumps({'variant':args.variant,'validated_seeds':validated,
                      'actor_robot_observation_dim':26,'action_dim':7,
                      'critic_or_GT_in_export':False},indent=2))


if __name__=='__main__':
    main()
