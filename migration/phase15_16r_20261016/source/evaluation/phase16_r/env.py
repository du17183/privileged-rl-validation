"""New protocol wrapper; all historical task and randomization files are read-only."""
import torch
from randomized_env.door_randomization import RandomizedDoorEnv
from door_env.door import make_cfg,door_reward,robot_observation,privileged_state
from environment_state.state_interface import get_environment_state

def unsafe(env):
    q=env.scene['robot'].data.joint_pos[:,:7]
    limits=env.scene['robot'].data.joint_pos_limits[:,:7]
    invalid=~torch.isfinite(q).all(-1)
    outside=((q<limits[:,:,0]-.001)|(q>limits[:,:,1]+.001)).any(-1)
    gt=privileged_state(env)
    return invalid|outside|~torch.isfinite(gt).all(-1)

class AtomicDoorEnv(RandomizedDoorEnv):
    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids):
            # Native scene reset writes q/qd but leaves articulation drive targets.
            # Clear those targets explicitly in this new protocol only.
            robot=self.scene['robot']
            robot.set_joint_position_target(robot.data.default_joint_pos[env_ids],env_ids=env_ids)
            robot.set_joint_velocity_target(robot.data.default_joint_vel[env_ids],env_ids=env_ids)
            robot.set_joint_effort_target(torch.zeros_like(robot.data.default_joint_pos[env_ids]),env_ids=env_ids)
            cabinet=self.scene['cabinet']
            # Keep the historical closed-door drive, not a sampled-angle hold.
            cabinet.set_joint_position_target(self.nominal_joints[env_ids],env_ids=env_ids)
            cabinet.set_joint_velocity_target(cabinet.data.default_joint_vel[env_ids],env_ids=env_ids)
            cabinet.set_joint_effort_target(torch.zeros_like(cabinet.data.default_joint_pos[env_ids]),env_ids=env_ids)
            self.scene.write_data_to_sim();self.sim.forward()

    def get_progress_state(self):
        state=get_environment_state(self)
        return dict(**state,handle_pose=torch.cat((state['handle_position'],state['handle_orientation']),-1))

def create(n,device,seed):
    from isaaclab.managers import RewardTermCfg,TerminationTermCfg
    cfg=make_cfg(n,device,seed)
    cfg.rewards.door=RewardTermCfg(func=door_reward,weight=1.)
    cfg.terminations.phase16_r_unsafe=TerminationTermCfg(func=unsafe)
    env=AtomicDoorEnv(cfg)
    env.set_randomization(2)
    env.reset(seed=seed);env.capture_terminal=True
    if abs(env.step_dt-1/60)>1e-9 or env.max_episode_length!=600:raise RuntimeError('Historical clock changed')
    return env
