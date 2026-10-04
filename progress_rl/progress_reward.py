"""Signed angle progress replaces the baseline positive-only velocity term."""
K_PROGRESS = 20.0
SUCCESS_RATE_REWARD = 600.0


def relabel(original, previous_angle, next_angle, next_velocity, dt):
    return original - K_PROGRESS * next_velocity.clamp_min(0) * dt + K_PROGRESS * (next_angle-previous_angle)


def reward(env):
    from door_env.door import door_reward, door_angle, door_velocity
    angle = door_angle(env).squeeze(-1)
    old = env.previous_angle.squeeze(-1)
    original = door_reward(env)
    return original - K_PROGRESS * door_velocity(env).squeeze(-1).clamp_min(0) + K_PROGRESS * (angle-old)/env.step_dt
