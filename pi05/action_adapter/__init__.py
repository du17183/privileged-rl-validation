"""Keep 7D relative EEF/gripper controller and original60Hz ticks."""
import numpy as np
HORIZON=10
EXECUTE=5
FLOW_STEPS=10

def chunk_indices(trajectory_index,horizon=HORIZON):
    # Padding repeats the last legal index, and is EXCLUDED by the loss mask.
    idx=np.arange(len(trajectory_index));last=np.empty_like(idx)
    edges=np.r_[np.flatnonzero(np.diff(trajectory_index))+1,len(idx)]
    start=0
    for end in edges:last[start:end]=end-1;start=end
    proposed=idx[:,None]+np.arange(horizon)[None]
    return np.minimum(proposed,last[:,None]),proposed<=last[:,None]

def to_environment(chunk):
    assert chunk.ndim==3 and chunk.shape[1:]==(HORIZON,32)
    if not np.isfinite(chunk).all():raise RuntimeError('Nonfinite generated actions')
    active=chunk[:,:,:7]
    return np.clip(active,-1,1).astype(np.float32),float(np.mean(np.abs(active)>1))
