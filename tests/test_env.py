import numpy as np
from gymnasium.utils.env_checker import check_env

from warehouse_env import WarehouseEnv


def test_env_checker_and_mask():
    env = WarehouseEnv(horizon=60)
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=3)
    assert observation["action_mask"].any()
    for _ in range(5):
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert sum(info["reward_terms"].values()) == reward
        assert observation["action_mask"].any()
        if terminated or truncated:
            break