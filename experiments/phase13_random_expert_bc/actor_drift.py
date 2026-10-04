"""Separate raw Gaussian-mean drift from bounded executed-action drift."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    choice=json.loads((R/'anchor_validation.json').read_text())
    anchor,mean,std,_=load_checkpoint(ROOT/choice['source'],'cpu')
    data=dict(np.load(ROOT/'datasets/random_door_expert/split_v1/test.npz'))
    rng=np.random.default_rng(14901);ids=rng.choice(len(data['robot']),2048,replace=False)
    x=inputs(torch.tensor(data['robot'][ids]),torch.tensor(data['environment'][ids]),mean,std,'B')
    reference_mu=anchor.mean(x);reference_action=reference_mu.tanh()
    rows=[]
    for seed in range(3):
        for arm in ['A','B','C']:
            policy,_,_,_=load_checkpoint(ROOT/f'checkpoints/phase13_random_expert_bc/rl/{arm}_seed{seed}/final.pt','cpu')
            mu=policy.mean(x);action=mu.tanh();delta=mu-reference_mu;adelta=action-reference_action
            rows.append(dict(seed=seed,arm=arm,raw_arm_rms=float(delta[:,:6].square().mean().sqrt()),
                        raw_gripper_rms=float(delta[:,6].square().mean().sqrt()),
                        action_arm_rms=float(adelta[:,:6].square().mean().sqrt()),
                        action_gripper_rms=float(adelta[:,6].square().mean().sqrt()),
                        gripper_binary_disagreement=float(((action[:,6]>0)!=(reference_action[:,6]>0)).float().mean()),
                        expert_state_normal_kl=float((.5*(delta/.01).square()).sum(-1).mean())))
    result=dict(rows=rows,reference_states=2048,split='offline test; no training or selection uses this analysis',
                source_fact='Existing Franka cabinet config uses BinaryJointPositionActionCfg for gripper; the arm has six relative Cartesian action channels.',
                source_file='third_party/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/joint_pos_env_cfg.py:39',
                interpretation='The Normal KL is mathematically a pre-tanh action-distribution diagnostic. Float saturation and binary executed gripper commands can make large raw/logit drift overstate mechanical action differences. No claim that KL alone explains failures.')
    (R/'actor_drift_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
