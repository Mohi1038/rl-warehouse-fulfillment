"""Rule-based policies. These are *evaluation-only* baselines, never the decision-maker.

All baselines read the same observation as the RL agents (no privileged state) and
share one dispatch template: optionally charge proactively, deliver when carried
items are urgent or nothing else fits, otherwise pick by the rule's priority.
"""

from __future__ import annotations

import numpy as np

from warehouse_env.env import WarehouseEnv

URGENT_TICKS = 45.0


class HeuristicPolicy:
    name = "heuristic"

    def __init__(self, env: WarehouseEnv, charge_threshold: float | None = None, seed: int = 0) -> None:
        self.n = env.max_orders
        self.docks = env.dock_count
        self.horizon = float(env.config.horizon)
        self.charge_action = env.charge_action
        self.wait_action = env.wait_action
        self.charge_threshold = charge_threshold
        self.rng = np.random.default_rng(seed)

    def priority(self, orders: np.ndarray, candidates: np.ndarray) -> int:
        raise NotImplementedError

    def act(self, obs: dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"].astype(bool)
        g = obs["global"]
        pick = np.flatnonzero(mask[: self.n])
        deliver = np.flatnonzero(mask[self.n : self.n + self.docks])
        if self.charge_threshold is not None and mask[self.charge_action] and g[3] < self.charge_threshold:
            return self.charge_action
        if deliver.size:
            time_left = g[10 : 10 + self.docks][deliver] * self.horizon
            urgent_dock = int(deliver[np.argmin(time_left)])
            if not pick.size or time_left.min() < URGENT_TICKS:
                return self.n + urgent_dock
        if pick.size:
            return int(self.priority(obs["orders"], pick))
        if mask[self.charge_action]:
            return self.charge_action
        return self.wait_action


class RandomPolicy(HeuristicPolicy):
    """Uniform over valid actions (no dispatch template)."""

    name = "random"

    def act(self, obs):
        return int(self.rng.choice(np.flatnonzero(obs["action_mask"])))


class FIFOPolicy(HeuristicPolicy):
    """Oldest order first."""

    name = "fifo"

    def priority(self, orders, candidates):
        return candidates[np.argmax(orders[candidates, 13])]


class EDFPolicy(HeuristicPolicy):
    """Earliest deadline (smallest slack) first."""

    name = "edf"

    def priority(self, orders, candidates):
        return candidates[np.argmin(orders[candidates, 8])]


class GreedyPolicy(HeuristicPolicy):
    """Highest value per unit distance: value / (dist + 1)."""

    name = "greedy"

    def priority(self, orders, candidates):
        score = orders[candidates, 1] / (orders[candidates, 11] + 1.0 / 40.0)
        return candidates[np.argmax(score)]


class EDFChargePolicy(EDFPolicy):
    """EDF plus a proactive charge-at-25% battery rule."""

    name = "edf_charge"

    def __init__(self, env, charge_threshold: float | None = 0.25, seed: int = 0):
        super().__init__(env, charge_threshold=charge_threshold, seed=seed)


BASELINES = {cls.name: cls for cls in (RandomPolicy, FIFOPolicy, EDFPolicy, GreedyPolicy, EDFChargePolicy)}


def make_baseline(name: str, env: WarehouseEnv, seed: int = 0) -> HeuristicPolicy:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    return BASELINES[name](env, seed=seed)
