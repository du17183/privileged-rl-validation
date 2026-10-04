"""KL of pre-tanh Gaussians; a common tanh transform preserves this KL."""
import math
import numpy as np
import torch


def normal_kl(mean, log_std, anchor_mean, anchor_log_std):
    return (anchor_log_std-log_std+.5*((2*(log_std-anchor_log_std)).exp()
            +(mean-anchor_mean).square()*(-2*anchor_log_std).exp()-1)).sum(-1)


def anchor_kl(actor, anchor, observation, effective=True):
    mean, log_std = actor.distribution(observation, effective)
    with torch.no_grad():
        ref_mean, ref_std = anchor.distribution(observation, effective)
    return normal_kl(mean, log_std, ref_mean, ref_std).mean()


@torch.no_grad()
def feasible_entropy_target(anchor, reference):
    """A cap-aware target avoids demanding entropy impossible under the bound.

    Deterministic quadrature estimates the capped anchor's squashed entropy.
    Keep target one nat below it, without drawing from the training RNG.
    This temperature feasibility adjustment is part of the std intervention.
    """
    mean, log_std = anchor.distribution(reference)
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    nodes = torch.as_tensor(nodes, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(weights/math.sqrt(math.pi), dtype=mean.dtype, device=mean.device)
    raw = mean[None]+math.sqrt(2)*log_std.exp()[None]*nodes[:, None, None]
    action = raw.tanh()
    log_jac = (1-action.square()+1e-6).log()
    entropy = (log_std+.5*math.log(2*math.pi*math.e)+(weights[:, None, None]*log_jac).sum(0)).sum(-1).mean()
    return min(-7.0, float(entropy)-1.0)

