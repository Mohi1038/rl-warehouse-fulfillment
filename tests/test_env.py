import gymnasium as gym
import numpy as np
import pytest
from conftest import rollout
from gymnasium.utils.env_checker import check_env

import warehouse_env  # noqa: F401  (registers the gymnasium ids)
from warehouse_env import MultiRobotWarehouseEnv, TinyWarehouseEnv, WarehouseEnv
from warehouse_env.env import CARRIED, CLAIMED, FULFILLED, LOADED, OPEN
from warehouse_env.tiny_env import hindsight_optimal_return

ENV_FACTORIES = {
    "single": lambda: WarehouseEnv(),
    "multi": lambda: MultiRobotWarehouseEnv(n_robots=3),
    "tiny": lambda: TinyWarehouseEnv(),
    "randomised": lambda: WarehouseEnv(layouts=("train_a", "train_b"), domain_randomization=True),
    "level3": lambda: _leveled(3),
}


def _leveled(level):
    env = WarehouseEnv()
    env.set_level(level)
    return env


@pytest.mark.parametrize("name", ENV_FACTORIES)
def test_gymnasium_env_checker(name):
    check_env(ENV_FACTORIES[name](), skip_render_check=True)


def test_registered_ids():
    env = gym.make("DockRush-v0")
    assert env.action_space.n == 21
    assert env.observation_space["orders"].shape == (16, 14)
    assert env.observation_space["global"].shape == (16,)


def _check_invariants(env, obs, reward, info, seen_fulfilled):
    assert obs["action_mask"].any(), "mask must never be all-false"
    assert obs["action_mask"][env.wait_action]
    assert sum(info["reward_terms"].values()) == reward
    assert np.isfinite(reward)
    claimed_by = {}
    for robot in env.robots:
        assert 0 <= robot.battery <= env.cfg.battery_capacity
        carried_weight = sum(order.weights[item] for order, item in robot.carried)
        pending = robot.option
        reserved = pending[1][0].weights[pending[1][1]] if pending and pending[0] == 0 else 0
        assert robot.load == carried_weight + reserved
        assert 0 <= robot.load <= env.capacity, "capacity exceeded"
        for order, item in robot.carried:
            assert order.item_state[item] == CARRIED
            assert (order.oid, item) not in claimed_by, "item held twice"
            claimed_by[(order.oid, item)] = robot.rid
    for order in env.orders:
        states = order.item_state
        if order.status == OPEN:
            assert order.loaded == states.count(LOADED)
            assert order.carried == states.count(CARRIED)
            assert order.unpicked == order.n - order.loaded - order.carried - states.count(CLAIMED)
        if order.status == FULFILLED:
            seen_fulfilled.setdefault(order.oid, env.time)
    assert len(seen_fulfilled) == env.stats["fulfilled"], "an order was fulfilled twice"
    slotted = [order for order in env.slots if order is not None]
    assert len({id(o) for o in slotted}) == len(slotted)
    assert all(o.status == OPEN and o.unpicked > 0 for o in slotted)


@pytest.mark.parametrize("name", ENV_FACTORIES)
@pytest.mark.parametrize("seed", range(4))
def test_invariants_hold_under_random_and_heuristic_play(name, seed, edf):
    env = ENV_FACTORIES[name]()
    for policy in (None, edf(env)):
        seen = {}
        rollout(env, policy, seed=seed, on_step=lambda e, o, r, i: _check_invariants(e, o, r, i, seen))


@pytest.mark.parametrize("name", ENV_FACTORIES)
def test_same_seed_gives_identical_trajectory(name):
    def signature(seed):
        trajectory, _ = rollout(ENV_FACTORIES[name](), seed=seed)
        return [(a, r, t, obs["global"].tobytes(), obs["orders"].tobytes()) for a, r, t, obs in trajectory]

    assert signature(5) == signature(5)
    assert signature(5) != signature(6)


def test_episode_resolves_every_order_and_terminates_at_horizon():
    env = WarehouseEnv()
    _, info = rollout(env, seed=1)
    assert env.time == env.horizon
    assert all(order.status != OPEN for order in env.orders)
    stats = info["episode_stats"]
    assert stats["orders"] == stats["fulfilled"] + stats["lost"] + stats["unfulfilled_closed"]
    assert 0 <= stats["on_time_pct"] <= 100


def test_options_are_temporally_extended():
    env = WarehouseEnv()
    obs, _ = env.reset(seed=0)
    _, _, _, _, info = env.step(env.wait_action)
    assert info["tau"] == env.cfg.wait_ticks
    while not obs["action_mask"][: env.max_orders].any():
        obs, *_ = env.step(env.wait_action)
    slot = int(np.flatnonzero(obs["action_mask"][: env.max_orders])[0])
    order, robot = env.slots[slot], env.robots[0]
    item = env._pick_item(robot, order)
    expected = env.layout.distances[robot.cell, order.access[item]] + env.cfg.pick_ticks
    _, _, _, _, info = env.step(slot)
    assert info["tau"] == expected
    assert robot.load == order.weights[item]


def test_invalid_action_falls_back_to_wait():
    env = WarehouseEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)  # no order in slot 0 at t=0
    assert info["tau"] == env.cfg.wait_ticks
    assert env.stats["invalid_actions"] == 1
    with pytest.raises(ValueError):
        env.step(99)


def test_battery_disabled_and_charging():
    env = WarehouseEnv(battery_enabled=False)
    obs, _ = env.reset(seed=0)
    assert not obs["action_mask"][env.charge_action] and obs["global"][3] == 1.0
    env = WarehouseEnv()
    env.reset(seed=0)
    env.robots[0].battery = 100.0
    env._observation()
    _, _, _, _, info = env.step(env.charge_action)
    assert env.robots[0].battery > 100.0 - info["tau"]
    assert env.robots[0].cell == env.layout.charger


def test_stranding_truncates_with_penalty():
    env = WarehouseEnv()
    env.reset(seed=0)
    env.robots[0].battery = 0.5
    env._observation()
    _, reward, terminated, truncated, info = env.step(env.wait_action)
    assert truncated and not terminated
    assert info["reward_terms"]["battery"] == -env.cfg.reward.lambda_battery
    assert info["episode_stats"]["battery_failures"] == 1


def test_shaping_is_potential_based():
    env = WarehouseEnv()
    obs, _ = env.reset(seed=2)
    gamma, alpha = env.cfg.gamma, env.cfg.reward.alpha
    done = False
    rng = np.random.default_rng(0)
    while not done:
        phi0 = env._potential()
        obs, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        done = terminated or truncated
        phi1 = 0.0 if done else env._potential()
        assert info["reward_terms"]["shaping"] == pytest.approx(alpha * (gamma ** info["tau"] * phi1 - phi0))


def test_backlog_beyond_slot_capacity():
    env = WarehouseEnv(base_rate=0.2, max_orders=4)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step(env.wait_action)
    assert all(order is not None for order in env.slots)
    assert obs["global"][13] > 0


def test_curriculum_levels_change_dynamics():
    easy, hard = _leveled(0), _leveled(3)
    easy.reset(seed=0)
    hard.reset(seed=0)
    assert not easy.battery_on and hard.battery_on
    assert len(easy.specs) < len(hard.specs)
    assert hard.layout.blocked


def test_larger_slot_counts_are_supported():
    env = WarehouseEnv(max_orders=32, base_rate=0.08)
    assert env.action_space.n == 32 + 3 + 2
    rollout(env, seed=0)


def test_multi_robot_claims_are_exclusive():
    env = MultiRobotWarehouseEnv(n_robots=2, base_rate=0.08)
    obs, _ = env.reset(seed=0)
    assert obs["robots"].shape == (3, 12) and obs["robots"][:2, 0].all() and not obs["robots"][2, 0]
    owners_seen = False
    rng = np.random.default_rng(0)
    for _ in range(300):
        me = env.active
        for index, order in enumerate(env.slots):
            if order is not None and order.owner not in (-1, me):
                owners_seen = True
                assert not obs["action_mask"][index]
        obs, _, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        if terminated or truncated:
            break
    assert owners_seen


def test_tiny_env_and_hindsight_oracle(edf):
    env = TinyWarehouseEnv()
    assert env.action_space.n == 3 + 1 + 2
    env.reset(seed=4)
    best = hindsight_optimal_return(env)
    env_copy = TinyWarehouseEnv(gamma=1.0)
    trajectory, _ = rollout(env_copy, edf(env_copy), seed=4)
    assert best >= sum(r for _, r, _, _ in trajectory) - 1e-6
