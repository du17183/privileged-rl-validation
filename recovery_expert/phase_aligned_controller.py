"""Frozen nominal BC+GT control head, after GT planner re-establishes grasp.

This is an explicit hybrid expert, not a pure IK-planner ablation. Only its
input clock is aligned to expert progress; physical episode time is intact.
All helper actions still execute through the original environment interface.
"""
from pathlib import Path
import json,h5py,numpy as np,torch
from door_env.door import robot_observation
from experiments.phase13_random_expert_bc.state import pack
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs


class PhaseAlignedController:
    def __init__(self,device):
        root=Path(__file__).resolve().parents[1]
        self.policy,self.mean,self.std,payload=load_checkpoint(root/'checkpoints/phase13_random_expert_bc/randomized_gt_candidate_anchor.pt',device)
        assert payload['arm']=='B'
        keys=json.loads((root/'datasets/random_door_expert/split_v1/split.json').read_text())['splits']['train']
        profile=[[] for _ in range(21)]
        with h5py.File(root/'datasets/random_door_expert/collection_v1/trajectories.h5','r') as h:
            for key in keys:
                g=h[key];phase=g['planner_phase'][:].flatten();ids=np.flatnonzero(phase==5)
                if not len(ids):continue
                angle=g['state'][:,0];clock=g['observation'][:,25]
                for b in range(21):profile[b].append(clock[ids[np.argmin(np.abs(angle[ids]-b*.05))]])
        self.clock=torch.tensor([float(np.median(v)) for v in profile],device=device)
        print('HYBRID_EXPERT frozen Phase13 nominal controller with train-only GT progress clock alignment',flush=True)

    @torch.no_grad()
    def action(self,env):
        robot=robot_observation(env).clone();context=pack(env.get_environment_state())
        robot[:,25]=self.clock[(context[:,0]*20).round().long().clamp(0,20)]
        action=self.policy(inputs(robot,context,self.mean,self.std,'B'));action[:,6]=-1
        return action.clamp(-1,1)
