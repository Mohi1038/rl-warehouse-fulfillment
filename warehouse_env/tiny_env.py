"""Tiny instance for the tabular Q-learning sanity check (PLAN.md §4.4)."""

from __future__ import annotations

import copy
import math

from .config import tiny_config
from .env import WarehouseEnv


class TinyWarehouseEnv(WarehouseEnv):
    """7x5 grid, at most 3 orders, capacity 2, one dock, no battery, no jitter.

    Same Dict observation and action semantics as the full environment, so PPO
    can be compared against tabular Q-learning on identical dynamics.
    """

    def __init__(self, render_mode: str | None = None, **overrides) -> None:
        super().__init__(tiny_config(**overrides), render_mode=render_mode)


def hindsight_optimal_return(env: WarehouseEnv, budget: int = 200_000) -> float:
    """Best undiscounted return achievable from ``env``'s current state with full knowledge of the future.

    An upper bound for any non-clairvoyant policy, computed by memoised search over a
    deep copy of the (deterministic once seeded) episode. Intended for tiny instances.
    """
    memo: dict = {}
    calls = 0

    def key(state: WarehouseEnv) -> tuple:
        robot = state.robots[state.active]
        orders = tuple((o.oid, o.slot, tuple(o.item_state), o.truck.truck_id) for o in state.open_orders)
        return (round(state.time, 6), robot.cell, state._next_spec, state._next_truck, orders)

    def search(state: WarehouseEnv) -> float:
        nonlocal calls
        calls += 1
        if calls > budget:
            raise RuntimeError("hindsight search budget exceeded")
        k = key(state)
        if k in memo:
            return memo[k]
        best = -math.inf
        for action in map(int, state._mask.nonzero()[0]):
            child = copy.deepcopy(state)
            _, reward, terminated, truncated, _ = child.step(action)
            value = reward + (0.0 if terminated or truncated else search(child))
            best = max(best, value)
        memo[k] = best
        return best

    snapshot = copy.deepcopy(env)
    snapshot.cfg = snapshot.cfg.replace(gamma=1.0)
    return search(snapshot)
