import numpy as np

from warehouse_env import MultiRobotWarehouseEnv, WarehouseEnv
from warehouse_env.render import record_gif, slack_color


def test_rgb_array_frames(tmp_path):
    env = WarehouseEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.set_attention(np.linspace(0, 1, 16))
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    rng = np.random.default_rng(0)
    frames = record_gif(env, lambda o: int(rng.choice(np.flatnonzero(o["action_mask"]))), str(tmp_path / "a.gif"), max_frames=5)
    assert frames == 6 and (tmp_path / "a.gif").stat().st_size > 0
    env.close()


def test_multi_robot_render():
    env = MultiRobotWarehouseEnv(n_robots=3, render_mode="rgb_array")
    env.reset(seed=0)
    assert env.render().shape[2] == 3


def test_slack_colors_go_from_red_to_green():
    assert slack_color(-10)[0] > slack_color(-10)[1]
    assert slack_color(500)[1] > slack_color(500)[0]
