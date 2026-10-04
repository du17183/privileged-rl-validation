# Physical fixture adapter contract

The simulated `PandaDrawerEnv.get_tool_state()` call is the fixture boundary. A physical adapter should return the same ordered 11 values, with synchronized timestamps: drawer displacement (m), drawer velocity (m/s), handle position (m) and quaternion (wxyz), and left/right finger-handle contact flags. Missing contact channels must be declared and handled by a new protocol; they must not be silently filled with simulator values.

The deployable A/B actor input is separate: nine Panda joint positions, nine velocities, TCP position and quaternion from robot kinematics, and elapsed episode time from the reset controller. It must not call the fixture adapter at inference. The B training critic may combine both streams after timestamp alignment. C intentionally consumes fixture state and remains a non-deployment upper bound.

The reset controller should close the drawer, return Panda to the defined home pose, clear contacts, and set the episode clock to zero. Begin the next episode only after measured drawer displacement and robot joints satisfy their reset tolerances. Preserve terminal `next_observation` and fixture `next_state` before reset, as `envs/isaac_env.py` does in simulation.

The current simulated command is 6-D relative Cartesian IK plus one gripper command at 60 Hz. A physical controller must map normalized commands to bounded Cartesian increments, enforce workspace/force limits, and log executed rather than requested actions when they differ. The replay schema remains `observation`, `state`, `action`, `reward`, `next_observation`, `next_state`, `done`, `terminated`, and `truncated`.

The simulator result is a feasibility test. Physical transfer still requires calibration of the drawer geometry, contact thresholds, actuation latency, camera or other perception if the fixture pose varies, and reset reliability. Those changes should be evaluated under a new protocol rather than mixed into the paired simulation results.
