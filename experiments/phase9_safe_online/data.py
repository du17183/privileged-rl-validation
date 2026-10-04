"""Read only successful frozen-anchor trajectories, retaining source identity."""
import h5py
from algorithms.replay import ReplayBuffer, FIELDS


def load_anchor(path, device):
    with h5py.File(path, "r") as h5:
        if h5.attrs["source"] != "frozen_phase8_anchor" or h5.attrs["optimizer_updates"] != 0:
            raise ValueError("Wrong trajectory source")
        groups = [g for g in h5.values() if g.attrs["success"] == 1]
        total = sum(len(g["action"]) for g in groups)
        if not groups:
            raise ValueError("No anchor successes")
        buffer = ReplayBuffer(total, 26, 11, 7, device)
        for group in groups:
            buffer.add({key: group[key][:] for key in FIELDS})
    return buffer

