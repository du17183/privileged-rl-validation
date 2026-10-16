"""Door-normal target, from the fixed physical door OBB frame."""
import numpy as np

def normal_target(planner,i,joint,distance=.08):
    pose=planner.fk(joint);center,normal=planner.door_frame(i)
    origin=planner.retreat_origins.setdefault(i,pose.translation.copy())
    return origin+distance*normal,normal,center
