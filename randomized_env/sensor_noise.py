"""Optional evaluation-only measurement corruption; disabled in core runs."""
import math
import torch


class SensorNoise:
    def __init__(self, device, seed, enabled=False):
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.enabled = enabled
        self.previous_contact = None

    def __call__(self, state):
        result = {k: v.clone() for k, v in state.items()}
        if not self.enabled:
            return result
        def uniform(shape, amplitude, device):
            return (2*torch.rand(shape, generator=self.rng, device=device)-1)*amplitude
        result['door_angle'] += uniform(result['door_angle'].shape, math.radians(.5), result['door_angle'].device)
        result['handle_position'] += uniform(result['handle_position'].shape, .002, result['handle_position'].device)
        contact = result['contact_state']
        delayed = self.previous_contact if self.previous_contact is not None else contact
        flip = torch.rand(contact.shape, generator=self.rng, device=contact.device) < .02
        result['contact_state'] = torch.where(flip, 1-delayed, delayed)
        self.previous_contact = contact.clone()
        # Progress is derived from the same noisy angle by callers; no clean GT
        # channel may silently bypass this sensor experiment.
        return result
