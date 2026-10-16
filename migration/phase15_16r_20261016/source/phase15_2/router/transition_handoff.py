"""A fifteen-tick, checked interpolation, with no repeated action padding."""
import numpy as np
def interpolate(normal,recovery,elapsed,ticks=15):
    alpha=np.clip((np.asarray(elapsed)+1)/ticks,0,1)
    return alpha[:,None]*normal+(1-alpha[:,None])*recovery
