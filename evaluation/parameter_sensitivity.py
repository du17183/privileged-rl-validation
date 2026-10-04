"""Offline interventions on identical robot state; no new training or rollout."""
import math
import torch


@torch.no_grad()
def sensitivity(actor, observations, arm):
    reference = actor(observations, deterministic=True)[0]
    result = {'reference_action_mean': reference.mean(0).tolist(), 'probes': []}
    if arm == 'A':
        result['probes'] = [{'feature': 'none', 'action_change_rms': 0.}]
        return result
    for coherent in (False, True):
        for degrees in (0., 2.5, 5.):
            altered = observations.clone();altered[:, 26] = math.radians(degrees)
            if coherent:
                # Counterfactual reset: initial angle equals measured angle, so
                # episode-relative progress is zero; target remains 1 rad.
                altered[:, 28] = 0.
            action = actor(altered, deterministic=True)[0]
            result['probes'].append(dict(feature='angle_coherent_reset' if coherent else 'angle_only', value_deg=degrees,
                action_mean=action.mean(0).tolist(), action_change_rms=float((action-reference).square().mean().sqrt()),
                xyz_action_change_rms=float((action[:, :3]-reference[:, :3]).square().mean().sqrt())))
    if arm in ('C', 'D'):
        for contact in (0., 1.):
            altered = observations.clone();altered[:, 29:31] = contact
            action = actor(altered, deterministic=True)[0]
            result['probes'].append(dict(feature='contact', value=contact, action_mean=action.mean(0).tolist(),
                action_change_rms=float((action-reference).square().mean().sqrt())))
    if arm == 'D':
        for axis in range(3):
            for offset in (-.01, .01):
                altered = observations.clone();altered[:, 31+axis] += offset/.1
                action = actor(altered, deterministic=True)[0]
                delta = action-reference
                result['probes'].append(dict(feature='handle_position', axis=axis, offset_m=offset,
                    action_mean=action.mean(0).tolist(), action_change_rms=float(delta.square().mean().sqrt()),
                    same_axis_action_delta=float(delta[:, axis].mean()),
                    same_axis_positive_alignment_fraction=float((delta[:, axis]*offset>0).float().mean())))
    result['caution'] = 'Sensitivity alone does not prove appropriate closed-loop adaptation; angle-only interventions can be inconsistent with pose/progress. Alignment is local action-space evidence, not full IK/contact proof.'
    return result
