import numpy as np
import pytest

from agents.baselines import make_baseline


def rollout(env, policy=None, seed=0, max_steps=10_000, on_step=None):
    """Run one episode; ``policy`` is a callable obs -> action (default: uniform over valid actions)."""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    trajectory = []
    for _ in range(max_steps):
        action = policy(obs) if policy else int(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((action, reward, info["tau"], obs))
        if on_step:
            on_step(env, obs, reward, info)
        if terminated or truncated:
            break
    return trajectory, info


@pytest.fixture
def edf():
    return lambda env: make_baseline("edf", env).act
