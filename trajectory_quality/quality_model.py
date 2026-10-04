"""Completed-trajectory physical score; never a policy or critic input.

This is an outcome-based replay heuristic, not a learned early success predictor.
Contact is measured only after actual opening starts, avoiding approach penalties.
"""
import numpy as np


def quality_score(success, final_angle, start_angle, target_angle, contact_stability):
    span = max(float(target_angle)-float(start_angle), 1e-6)
    final_progress = np.clip(float(final_angle)/max(float(target_angle), 1e-6), 0, 1)
    net_progress = np.clip((float(final_angle)-float(start_angle))/span, 0, 1)
    contact = np.clip(float(contact_stability), 0, 1)
    return float(.55*bool(success)+.20*final_progress+.15*net_progress+.10*contact)


def trajectory_score(state, next_state, success, target=1.0):
    start = float(state[0, 0])
    active = next_state[:, 0] > start+.02
    bilateral = (next_state[:, 9:11] > .5).all(axis=-1)
    contact = float(bilateral[active].mean()) if active.any() else 0.0
    score = quality_score(success, next_state[-1, 0], start, target, contact)
    return dict(quality_score=score, contact_stability=contact, start_angle=start,
                final_angle=float(next_state[-1, 0]), success=int(success))
