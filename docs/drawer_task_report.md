# Panda drawer task and experiment protocol

## Physical simulation

Isaac Sim 5.1 and Isaac Lab v2.3.0 supply the Franka Panda and Sektion cabinet in `Isaac-Open-Drawer-Franka-IK-Rel-v0`. The top drawer uses the translating `drawer_top_joint`. The 7-D normalized command contains six relative end-effector IK coordinates and one gripper command. The control period is 1/60 s. A trial succeeds when drawer displacement exceeds 0.30 m; success or the 8 s timeout triggers automatic reset.

The shared reward combines TCP proximity to the handle, bilateral finger-handle contact, positive drawer velocity, and a terminal opening bonus. Isaac Lab scales reward terms by the control period, so a terminal coefficient of 600 contributes 10 on the successful step. Evaluation logs drawer success and contact-assisted success separately.

## Observation boundary

The A/B deployable actor sees 26 robot-side values: nine Panda joint positions, nine joint velocities, TCP position and quaternion from robot kinematics, and normalized elapsed time from the reset controller. It has no cabinet-state or contact-sensor input. A's critic sees the same 26 values. B's critic also sees 11 fixture values: drawer position and velocity, handle position and quaternion, and left/right filtered finger-handle contact flags. C's actor and critic see both sets of values as a privileged-policy upper bound.

`PandaDrawerEnv.get_tool_state()` is the training-time fixture interface. Its output has the same role as future measurements from a physical reset jig. The A/B action path calls only `robot_observation()`; the GT call is confined to the replay collector, critic update, and evaluation diagnostics.

## Reset and state consistency

The stock randomized Panda joint-reset event is disabled. The preflight perturbed the drawer by 0.15 m and Panda's first joint by 0.05 rad; all 20 explicit reset trials returned closed drawer and Panda home. A separate forced-success step triggered automatic reset in all eight parallel environments. A no-action failure trial also timed out after 480 control steps and automatically restored closed drawer and Panda home. Immediately afterward, the TCP and handle frame poses exactly matched a fresh explicit reset. `PandaDrawerEnv` forwards PhysX during automatic reset before the next observation is read; contact flags are cleared until a new physics step. These checks cover simulation reset behavior, not a real fixture's recovery time.

## Expert data and shared interaction budget

The Cartesian waypoint planner reads the handle pose, approaches, aligns, closes the gripper, and pulls using differential IK. It generated 500 successful clean trajectories and 500 successful trajectories with Gaussian arm-command perturbation (standard deviation 0.08 in normalized action units) while correcting in closed loop. The two collections needed 1,012 attempted episodes and 308,256 simulator interactions, including failed and partial episodes. The merged HDF5 contains 1,000 successful trajectories and 300,645 saved transitions. Each transition has `observation`, `state`, `action`, `reward`, `next_observation`, `next_state`, `done`, `terminated`, and `truncated`. Terminal next states are captured before reset. See `docs/dataset_report_v2.md` and `datasets/drawer_expert_1000.h5`.

An exploratory DAgger-style collection produced 100,000 additional labeled interactions in `datasets/drawer_dagger_v1.h5`. It is **excluded** from the primary A/B/C experiment. When used in a supplementary experiment, its 100,000 interactions must be added to the conditioning budget.

## Primary paired protocol

`configs/experiment_primary.json` locks eight seeds (0–7), 32 parallel environments per run, 200,000 online SAC training interactions per variant, 64 deterministic evaluation episodes at step zero and every 10,000 training steps, 3,000 BC updates from the 1,000 demonstrations, and 1,000 critic-only offline updates. Online SAC uses four gradient updates per vector step, batch size 256, 25% demonstration replay, and a constant BC regularization weight of 10 for the actor. A, B, and C have identical settings and reward; only the specified observation routes differ. The main sample-efficiency axis counts online training interactions; threshold tables also include evaluation and the 308,256 shared demonstration-collection interactions.

Door articulation and DIVL are future extensions and are not part of this Drawer experiment.


