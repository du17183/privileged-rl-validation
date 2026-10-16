"""Adapt the state prefix on this instance; leave original π0.5 untouched."""
import types
from pi05.model import build, restore


def build_recovery(device='cuda:0', seed=0, initial=None):
    net = build(device=device, seed=seed)
    if initial is not None:
        restore(net, initial)

    def preprocess(self, observation, *, train=True):
        if observation.images or observation.state.shape[-1] != 47:
            raise RuntimeError('Recovery expects image-free 47-feature input')
        return [], [], observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state

    net._preprocess_observation = types.MethodType(preprocess, net)
    return net
