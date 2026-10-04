"""Physics/terminal/reward and matched network checks before formal rollout."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import hashlib
import json
from pathlib import Path
import torch
import isaaclab_tasks  # noqa
from door_env.door import door_angle, door_index
from door_env.isaac_env import transition_after_step
from offline_rl.baseline import load_demonstrations
from progress_rl.agent import make_agent, ExpertView
from progress_rl.door_env import config, create, live_observation, terminal_observation
from progress_rl.progress_state import observation
from progress_rl.progress_reward import relabel
ROOT = Path(__file__).resolve().parents[2]


def expected_base(robot, gt, dt):
    reach = 1-torch.tanh(5*torch.linalg.vector_norm(robot[:, 18:21]-gt[:, 2:5], dim=-1, keepdim=True))
    grasp = (gt[:, 9:11] > 0.5).all(dim=-1, keepdim=True).float()
    return dt*(reach+0.5*grasp+20*gt[:, 1:2].clamp_min(0)+600*(gt[:, :1] > 1).float())


def main():
    device = args.device or "cuda:0"
    env = create(config("D", 4, device, 9021))
    try:
        actor0 = make_agent("A", 0, device)
        actor1 = make_agent("C", 0, device)
        robot = live_observation(env, "A")
        gt = env.get_tool_state()
        augmented = observation(robot, gt, "C")
        a0 = actor0.act(robot, deterministic=True)
        a1 = actor1.act(augmented, deterministic=True)
        assert torch.allclose(a0, a1, atol=1e-6), "Initial Actor changed with zero feature weights"
        q0 = actor0.critic(robot, a0)
        q1 = actor1.critic(augmented, a1)
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(q0, q1)), "Initial critic mismatch"
        source = load_demonstrations(ROOT/"door_dataset"/"door_expert_1000.h5", env.device)
        batch = source.sample(4096)
        predicted = expected_base(batch["next_robot"], batch["next_privileged"], env.step_dt)
        residual = float((predicted-batch["reward"]).abs().max())
        if residual > 1e-3:
            raise RuntimeError(f"Expert reward scale/state mismatch: {residual}")
        ExpertView(source, "D", env.step_dt).sample(256)
        failures = []
        terminal_count = 0
        live_residual = 0.0
        for step in range(640):
            # Force an explicit success and verify the reset did not leak into next_state.
            if step == 10:
                cabinet = env.scene["cabinet"]
                pos = cabinet.data.joint_pos.clone()
                vel = cabinet.data.joint_vel.clone()
                pos[:, door_index(env)] = 1.1
                vel[:, door_index(env)] = 0
                cabinet.write_joint_state_to_sim(pos, vel)
                env.scene.write_data_to_sim()
                env.sim.forward()
            before = door_angle(env).clone()
            action = torch.zeros((env.num_envs, 7), device=env.device)
            _, reward, terminated, truncated, _ = env.step(action)
            next_robot, next_gt, done = transition_after_step(env, terminated, truncated)
            obs = terminal_observation(env, "D", done)
            expected = relabel(expected_base(next_robot, next_gt, env.step_dt), before,
                               next_gt[:, :1], next_gt[:, 1:2], env.step_dt)
            live_residual = max(live_residual, float((expected-reward[:, None]).abs().max()))
            assert torch.isfinite(obs).all()
            if done.any():
                terminal_count += int(done.sum())
                assert torch.equal(obs[done, 26:27], env.final_door_angle[done])
                assert (door_angle(env)[done] == 0).all(), "Automatic reset did not close door"
                assert torch.allclose(obs[done, 29:30], env.final_progress[done, 3:4])
        assert terminal_count >= 8, "Success/timeout terminal checks incomplete"
        assert live_residual < 1e-3, live_residual
        hashes = {}
        for name in ("door_dataset/door_expert_1000.h5", "door_env/door.py", "door_env/isaac_env.py"):
            h = hashlib.sha256()
            with (ROOT/name).open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    h.update(block)
            hashes[name] = h.hexdigest()
        result = dict(step_dt=env.step_dt, expert_reward_max_error=residual,
                      online_reward_max_error=live_residual, terminal_checks=terminal_count,
                      initial_actor_critic_matched=True, immutable_sha256=hashes)
        out = ROOT/"results"/"phase8_progress_rl"
        out.mkdir(parents=True, exist_ok=True)
        (out/"preflight_audit.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
