"""Load the executed distribution, including its cap, with robot-only inputs."""
from pathlib import Path
import torch
from safe_online.std_schedule import ControlledActor

class DeploymentPolicy:
    def __init__(self,path,device='cpu'):
        state=torch.load(Path(path),map_location=device,weights_only=False)
        if not state.get('validated_accepted') or state.get('robot_observation_dim')!=26:
            raise ValueError('Expected accepted Phase 9 deployment artifact')
        self.metadata={k:v for k,v in state.items() if k!='actor'}
        self.actor=ControlledActor().to(device).eval()
        self.actor.load_state_dict(state['actor'])
        self.actor.cap=state['std_cap']
        self.actor.requires_grad_(False)
        self.device=torch.device(device)

    @torch.no_grad()
    def action(self,robot_observation,stochastic=False):
        observation=torch.as_tensor(robot_observation,dtype=torch.float32,device=self.device)
        if observation.ndim!=2 or observation.shape[-1]!=26:
            raise ValueError('Supply N x 26 robot observations; no environment GT')
        return self.actor(observation,deterministic=not stochastic)[0]
