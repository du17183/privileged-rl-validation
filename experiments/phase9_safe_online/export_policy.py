"""Accepted policy artifacts, distribution metadata and inference consistency."""
import hashlib
import json
from pathlib import Path
import torch
from safe_online.deploy_policy import DeploymentPolicy
from safe_online.std_schedule import ControlledActor
from experiments.phase9_safe_online.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online/deployment'

def main():
    OUT.mkdir(exist_ok=False)
    records=[]
    for arm in ARMS:
        for seed in range(5):
            folder=ROOT/'checkpoints/phase9_safe_online'/f'P9{arm}_seed{seed}'
            for selection in ('best','final'):
                source=folder/('best.pt' if selection=='best' else 'step_300000.pt')
                state=torch.load(source,map_location='cpu',weights_only=False)
                artifact=dict(actor=state['actor'],std_cap=state['std_cap'],validated_accepted=True,
                    robot_observation_dim=26,action_dim=7,arm=arm,seed=seed,selection=selection,
                    additional_training_steps=state['env_steps'],is_initial_anchor=state['env_steps']==0,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    formal_validation=state['validation'],
                    anchor_sha256=state['anchor_sha256'],control='Unchanged relative IK scales xyz .05m/rotation .3rad, binary gripper; deployment policy consumes only existing robot observation')
                path=OUT/f'{arm}_seed{seed}_{selection}.pt'
                torch.save(artifact,path)
                deployed=DeploymentPolicy(path)
                original=ControlledActor();original.load_state_dict(state['actor']);original.cap=state['std_cap']
                torch.manual_seed(73453+seed)
                observations=torch.randn(128,26)
                assert torch.equal(original(observations,True)[0],deployed.action(observations))
                torch.manual_seed(93452+seed);expected=original(observations)[0]
                torch.manual_seed(93452+seed);actual=deployed.action(observations,stochastic=True)
                assert torch.equal(expected,actual)
                records.append(dict(arm=arm,seed=seed,selection=selection,steps=state['env_steps'],
                    std_cap=state['std_cap'],deterministic_and_sampling_bitwise_match=True,
                    path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'export_audit.json').write_text(json.dumps(records,indent=2))
    print('Exported and checked',len(records),'accepted policies')

if __name__=='__main__':main()
