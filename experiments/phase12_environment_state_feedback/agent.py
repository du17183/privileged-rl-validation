"""Function-preserving observation expansion, including Adam moments.

Actor and critic see exactly the same measured feature vector. New columns start
at zero; old weights, Q functions, targets, alpha and optimizer moments survive.
The frozen original anchor has zero new columns; it is deliberately not made
environment-aware by training, which would change the requested base algorithm.
"""
import copy
import torch
from algorithms.asymmetric_sac import TwinQ
from regularization.bc_regularized_sac import Phase6SAC
from safe_online.anchor_policy import AnchoredSAC
from safe_online.std_schedule import ControlledActor
from environment_feedback.normalization import DIMS


def expand_matrix(value, new_shape, critic=False):
    result = value.new_zeros(new_shape)
    result[:, :26] = value[:, :26]
    if critic:
        result[:, new_shape[1]-7:] = value[:, 26:]
    return result


def expanded_model(source, template, critic=False):
    result = {}
    for k, value in source.items():
        shape = template[k].shape
        result[k] = value.clone() if shape == value.shape else expand_matrix(value, shape, critic)
    return result


def expanded_optimizer(source, module, critic=False):
    result = copy.deepcopy(source)
    params = list(module.parameters())
    identifiers = [i for group in result['param_groups'] for i in group['params']]
    for identifier, parameter in zip(identifiers, params):
        for name, value in result['state'].get(identifier, {}).items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape != parameter.shape:
                result['state'][identifier][name] = expand_matrix(value, parameter.shape, critic)
    return result


class MeasuredSAC(AnchoredSAC):
    def __init__(self, checkpoint, anchor_checkpoint, arm, device, kl_weight=1.):
        dim = DIMS[arm]
        Phase6SAC.__init__(self, dim, 11, 7, device)
        self.arm = arm
        self.actor = ControlledActor(dim, 7).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.kl_weight = float(kl_weight)
        state = copy.deepcopy(checkpoint)
        for key in ('actor', 'critic', 'target_critic'):
            state[key] = expanded_model(checkpoint[key], getattr(self, key).state_dict(), key != 'actor')
        state['actor_opt'] = expanded_optimizer(checkpoint['actor_opt'], self.actor)
        state['critic_opt'] = expanded_optimizer(checkpoint['critic_opt'], self.critic, True)
        self.load_state(state)
        self.anchor = ControlledActor(dim, 7).to(self.device).eval()
        self.anchor.load_state_dict(expanded_model(anchor_checkpoint['actor'], self.anchor.state_dict()))
        self.actor.cap = self.anchor.cap = .01
        self.anchor.requires_grad_(False)

    def checkpoint(self):
        state = super().checkpoint()
        state.update(phase12_arm=self.arm, observation_dim=DIMS[self.arm], anchor=self.anchor.state_dict())
        return state
