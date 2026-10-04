"""Validate actual reset, signal interface, terminal handling and continuation."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args = parser.parse_args()
app = AppLauncher(headless=True).app
import json
import torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from safe_online.std_schedule import ControlledActor
from experiments.phase11_parameter_generalization.agent import MeasuredSAC
from experiments.phase11_parameter_generalization.data import MeasuredExpert, live_observation, transition
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, prepared
from environment_state.normalization import DIMS
from door_env.door import robot_observation
from algorithms.replay import ReplayBuffer, mixed_sample
try:
    protocol = prepared();env = create(32, args.device or 'cuda:0', 91000, 0)
    checks = []
    for seed in range(5):
        source = torch.load(ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{seed}'/'step_300000.pt', map_location=env.device, weights_only=False)
        anchor = torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{seed}'/'best.pt', map_location=env.device, weights_only=False)
        old_actor = ControlledActor().to(env.device);old_actor.load_state_dict(source['actor']);old_actor.cap = .01
        raw_robot = robot_observation(env).clone()
        agents = {}
        for arm in ('A', 'B', 'C', 'D'):
            agent = MeasuredSAC(source, anchor, arm, env.device)
            agents[arm] = agent
            obs = live_observation(env, arm, protocol['handle_center'])
            with torch.no_grad():
                old_mean, old_std = old_actor.distribution(raw_robot)
                mean, std = agent.actor.distribution(obs)
            if not torch.allclose(old_mean, mean, atol=2e-6) or not torch.equal(old_std, std): raise RuntimeError('Actor initial function changed')
            from algorithms.asymmetric_sac import TwinQ
            old_q = TwinQ(26, 7).to(env.device);old_q.load_state_dict(source['critic'])
            action = old_mean.tanh()
            for reference, expanded in zip(old_q(raw_robot, action), agent.critic(obs, action)):
                if not torch.allclose(reference, expanded, atol=3e-6): raise RuntimeError('Critic initial function changed')
            for name, parameter in agent.actor.named_parameters():
                old_value = source['actor'][name]
                if name != 'encoder.0.weight' and not torch.equal(parameter.detach(), old_value): raise RuntimeError('Old actor weights changed')
            checks.append(dict(seed=seed, arm=arm, actor_dim=DIMS[arm], critic_obs_dim=agent.critic_dim, unchanged_initial_policy=True))
        if seed == 0:
            env.set_randomization(2);env.reset(seed=91000)
            agent = agents['D']
            expert = MeasuredExpert(ROOT/'door_dataset/door_expert_1000.h5', env.device, env.step_dt, 'D', protocol['handle_center'])
            online = ReplayBuffer(64, DIMS['D'], 11, 7, env.device)
            obs = live_observation(env, 'D', protocol['handle_center']).clone();gt = env.get_tool_state().clone()
            with torch.no_grad(): action = agent.actor(obs)[0]
            _, reward, terminated, truncated, _ = env.step(action)
            nxt, next_gt, done = transition(env, terminated, truncated, 'D', protocol['handle_center'])
            online.add(dict(robot=obs, privileged=gt, action=action, reward=reward[:, None], next_robot=nxt, next_privileged=next_gt, done=done.float()[:, None]))
            result = agent.online_update(mixed_sample(online, expert, 256, .5), expert.sample(256), 10.)
            assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
            # Force timeout to verify terminal observation precedes automatic reset.
            before = env.get_environment_state()['door_angle'].clone()
            env.episode_length_buf[:] = env.max_episode_length
            _, _, term, trunc, _ = env.step(action)
            nxt, next_gt, done = transition(env, term, trunc, 'D', protocol['handle_center'])
            assert done.all() and torch.allclose(nxt[:, 26:27], next_gt[:, :1], atol=1e-7)
            assert torch.allclose(env.final_environment['door_angle'], next_gt[:, :1], atol=1e-7)
            assert torch.allclose(env.final_parameters[:, 0:1], env.final_start, atol=1e-5)
        env.set_randomization(0);env.reset(seed=91000)
    physical = []
    for level in range(4):
        env.set_randomization(level);env.reset(seed=91111)
        state = env.get_environment_state()
        physical.append(dict(level=level, angle_min_deg=float(state['door_angle'].min())*180/3.141592653589793,
            angle_max_deg=float(state['door_angle'].max())*180/3.141592653589793,
            offset_abs_max_m=float(env.parameters[:, 1:4].abs().max()), friction_min=float(env.parameters[:, 4].min()), friction_max=float(env.parameters[:, 4].max())))
    result = dict(passed=True, checks=checks, physical_randomization=physical,
        interface_fields=list(state), orientation_in_core_actor=False, terminal_snapshot_passed=True,
        optimizer_update_passed=True, physical_reset_checks=env.physical_reset_checks, smoke_interactions=64)
    (OUT/'preflight.json').write_text(json.dumps(result, indent=2));print(json.dumps(result), flush=True)
    env.close()
finally: app.close()
