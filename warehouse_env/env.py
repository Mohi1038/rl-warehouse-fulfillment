"""Event-driven single-robot warehouse environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .layout import Layout
from .orders import Order, OrderGenerator
from .reward import RewardTerms
from .trucks import Truck, TruckScheduler


@dataclass
class _LiveOrder:
    order: Order
    remaining: int
    carried: int = 0


class WarehouseEnv(gym.Env):
    """A bounded event-driven warehouse simulator suitable for RL agents."""

    metadata = {"render_modes": []}

    def __init__(self, layout_path: str | Path | None = None, horizon: int = 720, max_orders: int = 16, render_mode: str | None = None) -> None:
        super().__init__()
        self.layout = Layout.from_file(layout_path or Path(__file__).parent / "layouts" / "default.txt")
        self.horizon = horizon
        self.max_orders = max_orders
        self.render_mode = render_mode
        self.dock_count = len(self.layout.locations.get("D", ()))
        self.shelf_positions = self.layout.locations.get("S", ())
        self.charger = self.layout.locations["C"][0]
        self.packing = self.layout.locations["P"][0]
        self.action_space = spaces.Discrete(max_orders + self.dock_count + 2)
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
                "orders": spaces.Box(-np.inf, np.inf, shape=(max_orders, 14), dtype=np.float32),
                "action_mask": spaces.MultiBinary(int(self.action_space.n)),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        seed_value = int(self.np_random.integers(0, 2**32 - 1))
        self.time = 0.0
        self.robot = self.packing
        self.battery = 400.0
        self.capacity = 6
        self.carried: list[_LiveOrder] = []
        self.pending: list[_LiveOrder] = []
        self.completed: set[int] = set()
        self.orders = OrderGenerator(len(self.shelf_positions), self.dock_count, seed=seed_value, horizon=self.horizon).generate()
        self.trucks = TruckScheduler(self.dock_count, seed=seed_value).generate(self.horizon)
        self.next_order = 0
        self.next_truck = 0
        self._admit_arrivals()
        return self._observation(), {"tau": 0.0, "reward_terms": RewardTerms().as_dict(), "events": []}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        action = int(action)
        mask = self._action_mask()
        if not mask[action]:
            action = self.action_space.n - 1
        events: list[str] = []
        old_potential = self._potential()
        tau = self._duration(action)
        self._execute(action, events)
        self.time = min(float(self.horizon), self.time + tau)
        self.battery = max(0.0, self.battery - tau)
        self._admit_arrivals()
        fulfilled, lost, deferred = self._resolve_departures(events)
        stranded = self.battery <= 0
        terms = RewardTerms(
            fulfilled=sum(value for value in fulfilled),
            lost=-sum(value for value in lost),
            deferred=-0.3 * sum(value for value in deferred),
            operating_cost=-0.01 * tau,
            battery=-50.0 if stranded else 0.0,
            shaping=0.3 * (self._potential() - old_potential),
        )
        terminated = self.time >= self.horizon
        truncated = stranded and not terminated
        info = {"tau": tau, "reward_terms": terms.as_dict(), "events": events}
        return self._observation(), terms.total, terminated, truncated, info

    def _admit_arrivals(self) -> None:
        while self.next_order < len(self.orders) and self.orders[self.next_order].arrival_time <= self.time:
            order = self.orders[self.next_order]
            live = _LiveOrder(order, len(order.items))
            if len(self.pending) < self.max_orders:
                self.pending.append(live)
            self.next_order += 1

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        load = sum(order.order.weight * order.carried / len(order.order.items) for order in self.carried)
        for index, order in enumerate(self.pending[: self.max_orders]):
            if order.remaining and load + order.order.weight / len(order.order.items) <= self.capacity:
                mask[index] = 1
        for dock in range(self.dock_count):
            if any(order.order.dock == dock for order in self.carried):
                mask[self.max_orders + dock] = 1
        mask[self.max_orders + self.dock_count] = int(self.battery <= 380)
        mask[-1] = 1
        return mask

    def _duration(self, action: int) -> float:
        if action < self.max_orders and action < len(self.pending):
            order = self.pending[action]
            target = self.shelf_positions[order.order.items[len(order.order.items) - order.remaining]]
            return float(self.layout.distance(self.robot, target) + 2)
        if action < self.max_orders + self.dock_count:
            dock = action - self.max_orders
            return float(self.layout.distance(self.robot, self.layout.locations["D"][dock]) + 1)
        if action == self.max_orders + self.dock_count:
            return float(max(5, min(50, (400 - self.battery) / 4)))
        return 5.0

    def _execute(self, action: int, events: list[str]) -> None:
        if action < self.max_orders:
            order = self.pending[action]
            target = self.shelf_positions[order.order.items[len(order.order.items) - order.remaining]]
            self.robot = target
            order.remaining -= 1
            order.carried += 1
            if order.remaining == 0:
                self.pending.pop(action)
            if order not in self.carried:
                self.carried.append(order)
            events.append("item_picked")
        elif action < self.max_orders + self.dock_count:
            dock = action - self.max_orders
            self.robot = self.layout.locations["D"][dock]
            delivered = [order for order in self.carried if order.order.dock == dock]
            for order in delivered:
                self.carried.remove(order)
                self.completed.add(order.order.order_id)
            events.append(f"delivered:{len(delivered)}")
        elif action == self.max_orders + self.dock_count:
            self.robot = self.charger
            self.battery = min(400.0, self.battery + self._duration(action) * 4)
            events.append("charged")
        else:
            events.append("waited")

    def _resolve_departures(self, events: list[str]) -> tuple[list[float], list[float], list[float]]:
        fulfilled: list[float] = []
        lost: list[float] = []
        deferred: list[float] = []
        while self.next_truck < len(self.trucks) and self.trucks[self.next_truck].actual_departure <= self.time:
            truck = self.trucks[self.next_truck]
            for order in self.pending[:]:
                if order.order.dock != truck.dock or order.order.scheduled_departure > truck.scheduled_departure:
                    continue
                self.pending.remove(order)
                (lost if order.order.tier == "express" else deferred).append(order.order.value)
            self.next_truck += 1
            events.append(f"departure:{truck.truck_id}")
        return fulfilled, lost, deferred

    def _potential(self) -> float:
        return sum(order.order.value for order in self.carried)

    def _observation(self) -> dict[str, np.ndarray]:
        global_features = np.zeros(16, dtype=np.float32)
        global_features[:4] = (self.time / self.horizon, np.sin(2 * np.pi * self.time / self.horizon), np.cos(2 * np.pi * self.time / self.horizon), self.battery / 400)
        global_features[4] = sum(order.order.weight for order in self.carried) / self.capacity
        global_features[5:7] = np.asarray(self.robot, dtype=np.float32) / np.asarray(self.layout.grid.shape if hasattr(self.layout.grid, "shape") else (len(self.layout.grid), len(self.layout.grid[0])), dtype=np.float32)
        global_features[7] = len(self.pending)
        global_features[8] = len(self.orders) - self.next_order
        for dock in range(self.dock_count):
            global_features[9 + dock] = sum(order.carried for order in self.carried if order.order.dock == dock)
        orders = np.zeros((self.max_orders, 14), dtype=np.float32)
        for index, live in enumerate(self.pending[: self.max_orders]):
            orders[index, 0] = 1
            orders[index, 1] = live.order.value / 100
            orders[index, 2 + ("express", "standard", "bulk").index(live.order.tier)] = 1
            orders[index, 5 + live.order.dock] = 1
            orders[index, 8] = (live.order.scheduled_departure - self.time) / self.horizon
            orders[index, 9] = live.remaining
            orders[index, 10] = live.carried
            orders[index, 11] = live.order.weight <= self.capacity
            orders[index, 12] = (self.time - live.order.arrival_time) / self.horizon
        return {"global": global_features, "orders": orders, "action_mask": self._action_mask()}