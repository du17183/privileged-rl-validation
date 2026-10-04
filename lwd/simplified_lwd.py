"""Outcome-based selection only: no LWD networks, paper reproduction or DIVL."""
import torch


def selection_mask(scores, retained_fraction=.2):
    if not 0 < retained_fraction <= 1:
        raise ValueError("Invalid retained fraction")
    # Keep all boundary ties; never arbitrarily discard equally good successes.
    threshold = torch.quantile(scores, 1-retained_fraction)
    return scores >= threshold
