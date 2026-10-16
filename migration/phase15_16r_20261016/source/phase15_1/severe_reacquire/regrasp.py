import numpy as np
import pinocchio as pin
from phase15.reacquire.pregrasp_planner import rotation


def valid_grasp(planner,i,robot,gt,goal):
    pose=planner.fk(robot[:9]);position=float(np.linalg.norm(pose.translation-goal.translation))
    orientation=float(np.linalg.norm(pin.log3(goal.rotation@pose.rotation.T)))
    valid=bool(np.all(gt[9:11]>.5) and position<.025 and orientation<.20 and .005<robot[7:9].sum()<.079)
    return valid,dict(position_error_m=position,orientation_error_rad=orientation,
        bilateral_contact=bool(np.all(gt[9:11]>.5)),gripper_width_m=float(robot[7:9].sum()))
