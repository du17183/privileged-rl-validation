# Future extension plan

This plan is separate from the Drawer A/B/C experiment. Door, DIVL, and real robot control are not implemented in the current stage.

## Real fixture interface

The simulation environment exposes `get_privileged_state()` and the common call `get_tool_state()`. A real fixture driver should implement `get_tool_state()` with the same ordered fields and units: drawer displacement (m), drawer speed (m/s), handle position (m), handle quaternion (wxyz), and left/right contact flags. Each sample should carry a monotonic timestamp, reset generation, validity flag, and calibration version. Before a critic update, join the fixture sample with the robot observation and action from the same control cycle; reject stale or cross-reset pairs.

The A/B deployment actor receives only Panda joint position, joint velocity, gripper state, TCP pose, and elapsed time from the reset controller. The B critic may read fixture state during online training. The actor checkpoint contains no fixture input; verify this by running its inference API without a fixture connection. The C actor is an analysis upper bound and is not the deployment path.

The real reset controller should close the drawer, move Panda to its home configuration, confirm both with sensors, and only then begin a new episode. Log reset duration and failed restoration attempts separately from robot interaction steps. Enforce workspace, force, velocity, and emergency-stop limits in the robot controller, outside the policy.

## Door task after Drawer

Add a hinged door articulation and return door angle, angular velocity, handle pose, and contact state in the same fixture interface. Generate successful Panda door trajectories, then repeat the paired A/B/C protocol with unchanged success-evaluation accounting. Do not pool Door and Drawer results when making the first-stage claim.

## Value-guided replay after the base comparison

Keep uniform replay as the control. For a later DIVL-like study, compute trajectory value from a frozen critic, then change only the replay sampling probabilities. Compare uniform, value-weighted, and failure-weighted replay with matched seeds, online interaction budgets, demonstrations, BC initialization, and network capacity. Log sampling probabilities and effective sample size to catch replay collapse. QAM or VLA action-quality modeling is a later study.
