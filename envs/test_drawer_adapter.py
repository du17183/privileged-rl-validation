from types import SimpleNamespace

import torch

from drawer import drawer_reward, privileged_state, robot_observation, success


class Scene(dict):
    def __init__(self, *args, env_origins, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_origins = env_origins


def test_extraction():
    n = 2
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(n, 9), joint_vel=torch.zeros(n, 9)))
    cabinet = SimpleNamespace(
        joint_names=["drawer_top_joint"],
        data=SimpleNamespace(joint_pos=torch.tensor([[0.0], [0.31]]), joint_vel=torch.tensor([[0.0], [0.1]])),
    )
    ee = SimpleNamespace(data=SimpleNamespace(
        target_pos_w=torch.zeros(n, 1, 3), target_quat_w=torch.tensor([[[1., 0, 0, 0]], [[1., 0, 0, 0]]]),
    ))
    handle = SimpleNamespace(data=SimpleNamespace(
        target_pos_w=torch.ones(n, 1, 3), target_quat_w=torch.tensor([[[1., 0, 0, 0]], [[1., 0, 0, 0]]]),
    ))
    forces = torch.tensor([[[[1., 0, 0]]], [[[0., 0, 0]]]])
    left = SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces))
    right = SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces))
    scene = Scene(
        {"robot": robot, "cabinet": cabinet, "ee_frame": ee, "cabinet_frame": handle,
         "left_contact": left, "right_contact": right},
        env_origins=torch.zeros(n, 3),
    )
    env = SimpleNamespace(scene=scene, num_envs=n)
    assert robot_observation(env).shape == (2, 25)
    assert privileged_state(env).shape == (2, 11)
    assert success(env).tolist() == [False, True]
    assert torch.isfinite(drawer_reward(env)).all()


if __name__ == "__main__":
    test_extraction()
    print("Robot/GT split, contact flags, reward, success: PASS")
