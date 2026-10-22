"""Event-driven warehouse fulfilment environment (a semi-MDP with macro-actions).

The agent is queried only when a robot becomes free. Between decisions the clock
advances by ``τ`` ticks while orders arrive and trucks depart; ``info["tau"]`` holds
the option duration so learners can discount with ``γ^τ``. The simulator supports
``K`` robots natively (decisions are asynchronous: whichever robot frees up next
acts); ``WarehouseEnv`` with ``n_robots=1`` is the main single-robot task.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import CURRICULUM_LEVELS, EnvConfig
from .layout import Layout
from .orders import Order, OrderGenerator
from .reward import RewardAccumulator, total, zero_terms
from .trucks import Truck, TruckScheduler

GLOBAL_DIM = 16
ORDER_DIM = 14
ROBOT_DIM = 12
MAX_DOCKS = 3
MAX_ROBOTS = 3
OBS_BOUND = 10.0

OPEN, FULFILLED, LOST, CLOSED = range(4)
UNPICKED, CLAIMED, CARRIED, LOADED, DROPPED = range(5)
PICK, DELIVER, CHARGE, WAIT, IDLE = range(5)


class _LiveOrder:
    __slots__ = (
        "spec", "oid", "n", "access", "weights", "tier", "value", "dock", "truck", "item_state",
        "unpicked", "carried", "loaded", "owner", "status", "deferrals", "slot", "loaded_at",
    )  # fmt: skip

    def __init__(self, spec: Order, access: np.ndarray, truck: Truck) -> None:
        self.spec = spec
        self.oid = spec.order_id
        self.n = len(spec.items)
        self.access = [int(access[item]) for item in spec.items]
        self.weights = list(spec.item_weights)
        self.tier = spec.tier
        self.value = spec.value
        self.dock = spec.dock
        self.truck = truck
        self.item_state = [UNPICKED] * self.n
        self.unpicked = self.n
        self.carried = 0
        self.loaded = 0
        self.owner = -1
        self.status = OPEN
        self.deferrals = 0
        self.slot = -1
        self.loaded_at = -1.0


class _Robot:
    __slots__ = ("rid", "cell", "battery", "load", "carried", "busy_until", "option", "target", "stranded")

    def __init__(self, rid: int, cell: int, battery: float) -> None:
        self.rid = rid
        self.cell = cell
        self.battery = battery
        self.load = 0
        self.carried: list[tuple[_LiveOrder, int]] = []
        self.busy_until = 0.0
        self.option: tuple | None = None
        self.target = cell
        self.stranded = False


class WarehouseEnv(gym.Env):
    """DockRush: deadline-aware order fulfilment with truck departures, capacity and battery limits.

    Actions (``Discrete(N_max + D + 2)``): ``PICK(i)`` for order slot ``i``, ``DELIVER(d)``
    for dock ``d``, ``CHARGE`` and ``WAIT``. Invalid actions are replaced by ``WAIT``;
    agents should use ``obs["action_mask"]``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, config: EnvConfig | dict | None = None, render_mode: str | None = None, **overrides: Any) -> None:
        super().__init__()
        if isinstance(config, dict) or config is None:
            config = EnvConfig.from_dict({**(config or {}), **overrides})
        elif overrides:
            config = config.replace(**overrides)
        self.config = config
        self.render_mode = render_mode
        self.level: int | None = None
        self.max_orders = config.max_orders
        self.n_robots = config.n_robots
        if not 1 <= self.n_robots <= MAX_ROBOTS:
            raise ValueError(f"n_robots must be in [1, {MAX_ROBOTS}]")
        layouts = [Layout.named(name) for name in config.layouts]
        dock_counts = {len(layout.docks) for layout in layouts}
        if len(dock_counts) != 1:
            raise ValueError("all layouts in a config must have the same number of docks")
        self.dock_count = dock_counts.pop()
        self.n_actions = self.max_orders + self.dock_count + 2
        self.charge_action = self.max_orders + self.dock_count
        self.wait_action = self.charge_action + 1
        self.action_space = spaces.Discrete(self.n_actions)
        obs_spaces = {
            "global": spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(GLOBAL_DIM,), dtype=np.float32),
            "orders": spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(self.max_orders, ORDER_DIM), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
        }
        if config.include_robot_tokens or self.n_robots > 1:
            obs_spaces["robots"] = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(MAX_ROBOTS, ROBOT_DIM), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)
        self.attention: np.ndarray | None = None
        self._renderer = None
        self._layout_cache: dict[str, Layout] = {}
        self.layout = layouts[0]

    # ------------------------------------------------------------------ public API
    def set_level(self, level: int | None) -> None:
        """Select a curriculum level (PLAN.md §5.1); ``None`` uses the raw config. Applies on next reset."""
        if level is not None and not 0 <= level < len(CURRICULUM_LEVELS):
            raise ValueError(f"curriculum level must be in [0, {len(CURRICULUM_LEVELS)})")
        self.level = level

    def set_attention(self, weights: np.ndarray | None) -> None:
        """Attention over order slots, drawn by the renderer as an overlay."""
        self.attention = None if weights is None else np.asarray(weights, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self.config
        if self.level is not None:
            cfg = cfg.replace(**CURRICULUM_LEVELS[self.level])
        if options and "config" in options:
            cfg = cfg.replace(**options["config"])
        if cfg.domain_randomization:
            cfg = cfg.replace(
                lambda_mult=cfg.lambda_mult * float(rng.uniform(*cfg.dr_lambda_range)),
                capacity=int(rng.integers(cfg.dr_capacity_range[0], cfg.dr_capacity_range[1] + 1)),
                jitter_std=float(rng.uniform(*cfg.dr_jitter_range)),
            )
        self.cfg = cfg
        self.layout = self._pick_layout(cfg, rng)
        self.horizon = float(cfg.horizon)
        self.capacity = cfg.capacity
        self.battery_on = cfg.battery_enabled and self.layout.charger is not None
        self._hop = self._mean_shelf_hop(self.layout)
        self._max_dist = max(1, self.layout.max_distance)

        self.trucks = TruckScheduler(
            self.dock_count, cfg.truck_interval, cfg.jitter_std, seed=rng, first_range=cfg.first_truck_range
        ).generate(cfg.horizon)
        self._dock_trucks = [[t for t in self.trucks if t.dock == d] for d in range(self.dock_count)]
        for trucks in self._dock_trucks:
            trucks.sort(key=lambda t: t.scheduled_departure)
        self._dock_next = [0] * self.dock_count
        self.specs = OrderGenerator(
            len(self.layout.shelves),
            self.dock_count,
            base_rate=cfg.base_rate * cfg.lambda_mult,
            horizon=cfg.horizon,
            seed=rng,
            rate_amplitude=cfg.rate_amplitude,
            spike_height=cfg.spike_height,
            spike_width=cfg.spike_width,
            items_range=cfg.items_range,
            item_weight_range=cfg.item_weight_range,
            tier_probs=cfg.tier_probs,
            value_mu=cfg.value_mu,
            value_sigma=cfg.value_sigma,
            zipf_exponent=cfg.zipf_exponent,
            vip_prob=cfg.vip_prob if cfg.shocks_enabled else 0.0,
            vip_value_mult=cfg.vip_value_mult,
            max_orders=cfg.max_total_orders,
        ).generate()

        self.time = 0.0
        self.robots = [_Robot(i, self.layout.packing, cfg.battery_capacity) for i in range(self.n_robots)]
        self.active = 0
        self.orders: list[_LiveOrder] = []
        self.open_orders: list[_LiveOrder] = []
        self.slots: list[_LiveOrder | None] = [None] * self.max_orders
        self.backlog: deque[_LiveOrder] = deque()
        self._next_spec = 0
        self._next_truck = 0
        self._recent_arrivals: deque[float] = deque()
        self._acc = RewardAccumulator()
        self._events: list[str] = []
        self.stats = _new_stats()
        self.episode_return = 0.0
        self.attention = None
        self._advance()
        self._events = []
        obs = self._observation()
        return obs, {"tau": 0.0, "reward_terms": zero_terms(), "events": [], "robot": self.active}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")
        action = int(action)
        robot = self.robots[self.active]
        if not self._mask[action]:
            self.stats["invalid_actions"] += 1
            action = self.wait_action
        self._acc = RewardAccumulator()
        self._events = []
        t0 = self.time
        phi0 = self._potential()
        self._start_option(robot, action)
        stranded = self._advance()
        terminated = self.time >= self.horizon
        truncated = stranded and not terminated
        tau = self.time - t0
        phi1 = 0.0 if (terminated or truncated) else self._potential()
        shaping = self.cfg.reward.alpha * (self.cfg.gamma**tau * phi1 - phi0)
        terms = self._acc.terms(self.cfg.reward, tau, self.n_robots, stranded, shaping)
        reward = total(terms)
        self.episode_return += reward
        self.stats["decisions"] += 1
        self.stats["action_counts"][self._action_kind(action)] += 1
        obs = self._observation()
        info = {"tau": tau, "reward_terms": terms, "events": self._events, "robot": self.active}
        if terminated or truncated:
            info["episode_stats"] = self.episode_stats()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions for the robot that is about to act (sb3-contrib compatible)."""
        return self._mask.astype(bool)

    def episode_stats(self) -> dict[str, float]:
        s = self.stats
        resolved = max(1, s["orders"])
        express = max(1, s["express_orders"])
        fulfilled = max(1, s["fulfilled"])
        return {
            "return": self.episode_return,
            "orders": s["orders"],
            "fulfilled": s["fulfilled"],
            "lost": s["lost"],
            "deferred": s["deferred"],
            "unfulfilled_closed": s["closed"],
            "revenue_captured_pct": 100.0 * s["revenue_captured"] / max(1e-9, s["revenue_total"]),
            "on_time_pct": 100.0 * s["on_time"] / resolved,
            "express_on_time_pct": 100.0 * s["express_on_time"] / express,
            "trucks_missed": s["trucks_missed"],
            "mean_slack_at_delivery": s["slack_sum"] / fulfilled,
            "battery_failures": s["battery_failures"],
            "travel_ticks_per_order": s["travel_ticks"] / fulfilled,
            "items_per_trip": s["items_delivered"] / max(1, s["deliveries"]),
            "invalid_actions": s["invalid_actions"],
            "decisions": s["decisions"],
            "level": -1 if self.level is None else self.level,
        }

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .render import Renderer

            self._renderer = Renderer(self)
        return self._renderer.draw(self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def tabular_state(self) -> tuple:
        """Exact hashable state for tabular methods on small instances (see ``tiny_env``)."""
        robot = self.robots[self.active]
        orders = tuple(
            (o.slot, o.tier, o.unpicked, o.carried, o.loaded, int(o.truck.scheduled_departure))
            for o in self.open_orders
        )
        return (robot.cell, int(self.time // 10), orders)

    # ------------------------------------------------------------------ options
    def _start_option(self, robot: _Robot, action: int) -> None:
        cfg = self.cfg
        dist = self.layout.distances
        t = self.time
        if action < self.max_orders:
            order = self.slots[action]
            item = self._pick_item(robot, order)
            cell = order.access[item]
            travel = int(dist[robot.cell, cell])
            duration = travel + cfg.pick_ticks
            drain = travel * cfg.move_drain + cfg.pick_ticks * cfg.idle_drain
            order.item_state[item] = CLAIMED
            order.unpicked -= 1
            robot.load += order.weights[item]
            if self.n_robots > 1:
                order.owner = robot.rid
            robot.option = (PICK, (order, item), cell, drain)
            self._events.append(f"pick:{order.oid}")
            self._refresh_slots()
        elif action < self.charge_action:
            dock = action - self.max_orders
            cell = self.layout.docks[dock]
            travel = int(dist[robot.cell, cell])
            n_items = sum(1 for order, _ in robot.carried if order.dock == dock)
            duration = travel + n_items * cfg.load_ticks_per_item
            drain = travel * cfg.move_drain + n_items * cfg.load_ticks_per_item * cfg.idle_drain
            robot.option = (DELIVER, dock, cell, drain)
        elif action == self.charge_action:
            cell = self.layout.charger
            travel = int(dist[robot.cell, cell])
            arrive = t + travel
            on_arrival = robot.battery - travel * cfg.move_drain
            need = max(1, math.ceil((cfg.battery_capacity - on_arrival) / cfg.charge_rate))
            next_arrival = self._next_arrival_after(arrive)
            ticks = need
            if next_arrival is not None and next_arrival < arrive + need:
                ticks = max(min(cfg.wait_ticks, need), math.ceil(next_arrival - arrive))
            duration = travel + ticks
            drain = travel * cfg.move_drain - ticks * cfg.charge_rate
            robot.option = (CHARGE, ticks, cell, drain)
            self._events.append("charge")
        else:
            cell = robot.cell
            duration = cfg.wait_ticks
            drain = cfg.wait_ticks * cfg.idle_drain
            robot.option = (WAIT, None, cell, drain)
        robot.target = cell
        robot.busy_until = t + duration
        if self.battery_on:
            travel_drain = (
                drain if robot.option[0] != CHARGE else int(dist[robot.cell, cell]) * cfg.move_drain
            )
            if robot.battery - travel_drain < 0:
                robot.stranded = True
                robot.busy_until = t + robot.battery / max(cfg.move_drain, cfg.idle_drain)

    def _complete(self, robot: _Robot) -> None:
        option, robot.option = robot.option, None
        if option is None:
            return
        kind, payload, cell, drain = option
        cfg = self.cfg
        if robot.stranded:
            robot.battery = 0.0
            self.stats["battery_failures"] += 1
            self._events.append(f"stranded:{robot.rid}")
            return
        self.stats["travel_ticks"] += int(self.layout.distances[robot.cell, cell])
        robot.cell = robot.target = cell
        if self.battery_on:
            robot.battery = min(cfg.battery_capacity, robot.battery - drain)
        if kind == PICK:
            order, item = payload
            if order.status == OPEN and order.item_state[item] == CLAIMED:
                order.item_state[item] = CARRIED
                order.carried += 1
                robot.carried.append((order, item))
            else:
                robot.load -= order.weights[item]
        elif kind == DELIVER:
            keep, delivered = [], 0
            for order, item in robot.carried:
                if order.dock != payload:
                    keep.append((order, item))
                    continue
                order.item_state[item] = LOADED
                order.carried -= 1
                order.loaded += 1
                robot.load -= order.weights[item]
                delivered += 1
                if order.loaded == order.n:
                    order.loaded_at = self.time
            robot.carried = keep
            self.stats["deliveries"] += 1
            self.stats["items_delivered"] += delivered
            self._events.append(f"deliver:{payload}:{delivered}")

    def _pick_item(self, robot: _Robot, order: _LiveOrder) -> int:
        """Nearest unclaimed item of ``order`` that fits the remaining capacity, or -1."""
        best, best_dist = -1, 1 << 30
        free = self.capacity - robot.load
        row = self.layout.distances[robot.cell]
        for item in range(order.n):
            if order.item_state[item] == UNPICKED and order.weights[item] <= free:
                d = row[order.access[item]]
                if d < best_dist:
                    best, best_dist = item, d
        return best

    # ------------------------------------------------------------------ events
    def _advance(self) -> bool:
        """Run the event queue until a robot is free to decide. Returns True if a robot stranded."""
        robot = min(self.robots, key=lambda r: (r.busy_until, r.rid))
        end = min(robot.busy_until, self.horizon)
        self._process_events(end, include_end=end >= self.horizon)
        self.time = end
        if robot.busy_until >= self.horizon:
            return False
        self._complete(robot)
        self.active = robot.rid
        return robot.stranded

    def _process_events(self, end: float, include_end: bool) -> None:
        specs, trucks = self.specs, self.trucks
        while True:
            next_arrival = specs[self._next_spec].arrival_time if self._next_spec < len(specs) else math.inf
            next_departure = trucks[self._next_truck].actual_departure if self._next_truck < len(trucks) else math.inf
            if next_arrival <= next_departure and next_arrival <= end:
                self.time = next_arrival
                self._arrive(specs[self._next_spec])
                self._next_spec += 1
            elif next_departure < end or (include_end and next_departure <= end):
                self.time = next_departure
                self._depart(trucks[self._next_truck])
                self._next_truck += 1
            else:
                return

    def _arrive(self, spec: Order) -> None:
        cfg = self.cfg
        slack = (cfg.vip_min_slack if spec.vip else cfg.min_slack[spec.tier]) * cfg.slack_mult
        order = _LiveOrder(spec, self.layout.shelf_access, self._truck_for(spec.dock, spec.arrival_time + slack))
        self.orders.append(order)
        self.open_orders.append(order)
        self.backlog.append(order)
        self._recent_arrivals.append(spec.arrival_time)
        self.stats["orders"] += 1
        self.stats["revenue_total"] += spec.value
        self.stats["express_orders"] += spec.tier == 0
        self._events.append(f"arrival:{spec.order_id}")
        self._refresh_slots()

    def _depart(self, truck: Truck) -> None:
        reward = self.cfg.reward
        missed = False
        fulfilled = lost = deferred = 0
        self._dock_next[truck.dock] += 1
        for order in [o for o in self.open_orders if o.truck is truck]:
            if order.loaded == order.n:
                order.status = FULFILLED
                self._acc.fulfilled += order.value * reward.tier_bonus[order.tier]
                self.stats["fulfilled"] += 1
                self.stats["revenue_captured"] += order.value
                self.stats["slack_sum"] += truck.scheduled_departure - order.loaded_at
                if order.deferrals == 0:
                    self.stats["on_time"] += 1
                    self.stats["express_on_time"] += order.tier == 0
                fulfilled += 1
                continue
            missed = True
            nxt = self._truck_after(truck)
            if order.tier == 0 or nxt is None:
                self._close(order, LOST if order.tier == 0 else CLOSED)
                if order.tier == 0:
                    self._acc.lost += order.value
                    self.stats["lost"] += 1
                    lost += 1
                else:
                    self._acc.deferred += order.value
                    self.stats["closed"] += 1
                    deferred += 1
            else:
                self._acc.deferred += order.value
                order.value *= 0.5
                order.truck = nxt
                order.deferrals += 1
                self.stats["deferred"] += 1
                deferred += 1
        self.open_orders = [o for o in self.open_orders if o.status == OPEN]
        self.stats["trucks_missed"] += missed
        self._events.append(f"departure:{truck.truck_id}:{fulfilled}:{lost}:{deferred}")
        self._refresh_slots()

    def _close(self, order: _LiveOrder, status: int) -> None:
        order.status = status
        for robot in self.robots:
            if order.carried:
                kept = []
                for entry in robot.carried:
                    if entry[0] is order:
                        robot.load -= order.weights[entry[1]]
                    else:
                        kept.append(entry)
                robot.carried = kept
        for item, state in enumerate(order.item_state):
            if state in (CLAIMED, CARRIED, UNPICKED):
                order.item_state[item] = DROPPED
        order.carried = 0

    def _truck_for(self, dock: int, earliest: float) -> Truck:
        for truck in self._dock_trucks[dock][self._dock_next[dock] :]:
            if truck.scheduled_departure >= earliest:
                return truck
        return self._dock_trucks[dock][-1]

    def _truck_after(self, truck: Truck) -> Truck | None:
        trucks = self._dock_trucks[truck.dock]
        index = trucks.index(truck)
        return trucks[index + 1] if index + 1 < len(trucks) else None

    def _next_arrival_after(self, time: float) -> float | None:
        for spec in self.specs[self._next_spec :]:
            if spec.arrival_time > time:
                return spec.arrival_time
        return None

    def _refresh_slots(self) -> None:
        slots = self.slots
        for index, order in enumerate(slots):
            if order is not None and (order.status != OPEN or order.unpicked == 0):
                order.slot = -1
                slots[index] = None
        if None not in slots:
            return
        free = [index for index, order in enumerate(slots) if order is None]
        while free and self.backlog:
            order = self.backlog.popleft()
            if order.status != OPEN or order.unpicked == 0:
                continue
            index = free.pop(0)
            order.slot = index
            slots[index] = order

    # ------------------------------------------------------------------ observation & mask
    def _potential(self) -> float:
        """Φ(s): value of carried or dock-loaded items whose truck is still reachable."""
        dist = self.layout.distances
        docks = self.layout.docks
        phi = 0.0
        for order in self.open_orders:
            if order.loaded:
                phi += order.value * order.loaded / order.n
        for robot in self.robots:
            for order, _ in robot.carried:
                if self.time + dist[robot.cell, docks[order.dock]] <= order.truck.scheduled_departure:
                    phi += order.value / order.n
        return phi

    def _observation(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        layout = self.layout
        dist = layout.distances
        robot = self.robots[self.active]
        t, horizon = self.time, self.horizon
        rows, cols = layout.shape
        position = layout.positions[robot.cell]
        mask = np.zeros(self.n_actions, dtype=np.int8)
        battery_floor = self._battery_floor(robot)

        g = np.zeros(GLOBAL_DIM, dtype=np.float32)
        phase = 2 * math.pi * t / horizon
        g[0] = t / horizon
        g[1] = math.sin(phase)
        g[2] = math.cos(phase)
        g[3] = robot.battery / cfg.battery_capacity if self.battery_on else 1.0
        g[4] = robot.load / self.capacity
        g[5] = position[0] / rows
        g[6] = position[1] / cols
        carried_per_dock = [0] * self.dock_count
        for order, _ in robot.carried:
            carried_per_dock[order.dock] += 1
        for dock in range(self.dock_count):
            g[7 + dock] = carried_per_dock[dock] / self.capacity
            trucks = self._dock_trucks[dock]
            nxt = self._dock_next[dock]
            if nxt < len(trucks):
                g[10 + dock] = min(1.0, max(0.0, (trucks[nxt].scheduled_departure - t) / horizon))
            if carried_per_dock[dock] and self._reachable(robot, layout.docks[dock], battery_floor):
                mask[self.max_orders + dock] = 1
        g[13] = sum(1 for o in self.backlog if o.status == OPEN and o.unpicked) / self.max_orders
        while self._recent_arrivals and self._recent_arrivals[0] < t - 60:
            self._recent_arrivals.popleft()
        g[14] = len(self._recent_arrivals) / 60.0 / (2 * self.config.base_rate)
        g[15] = self.capacity / 10.0

        orders = np.zeros((self.max_orders, ORDER_DIM), dtype=np.float32)
        free = self.capacity - robot.load
        row = dist[robot.cell]
        for index, order in enumerate(self.slots):
            if order is None:
                continue
            nearest, nearest_dist, fits = -1, 1 << 30, -1
            fits_dist = 1 << 30
            for item in range(order.n):
                if order.item_state[item] != UNPICKED:
                    continue
                d = int(row[order.access[item]])
                if d < nearest_dist:
                    nearest, nearest_dist = item, d
                if order.weights[item] <= free and d < fits_dist:
                    fits, fits_dist = item, d
            remaining = order.unpicked
            est = (
                nearest_dist
                + cfg.pick_ticks * remaining
                + (remaining - 1) * self._hop
                + dist[order.access[nearest], layout.docks[order.dock]]
                + order.n * cfg.load_ticks_per_item
            )
            features = orders[index]
            features[0] = 1.0
            features[1] = order.value / 50.0
            features[2 + order.tier] = 1.0
            features[5 + order.dock] = 1.0
            features[8] = (order.truck.scheduled_departure - t - est) / horizon
            features[9] = remaining / 4.0
            features[10] = (order.carried + order.loaded) / 4.0
            features[11] = nearest_dist / self._max_dist
            features[12] = float(fits >= 0)
            features[13] = (t - order.spec.arrival_time) / horizon
            owned_elsewhere = order.owner not in (-1, robot.rid)
            if fits >= 0 and not owned_elsewhere and self._reachable(robot, order.access[fits], battery_floor, cfg.pick_ticks):
                mask[index] = 1
        if self.battery_on and robot.battery <= 0.95 * cfg.battery_capacity:
            mask[self.charge_action] = 1
        mask[self.wait_action] = 1
        self._mask = mask
        np.clip(orders, -OBS_BOUND, OBS_BOUND, out=orders)
        obs = {"global": g, "orders": orders, "action_mask": mask.copy()}
        if "robots" in self.observation_space.spaces:
            obs["robots"] = self._robot_tokens(robot)
        return obs

    def _robot_tokens(self, me: _Robot) -> np.ndarray:
        rows, cols = self.layout.shape
        tokens = np.zeros((MAX_ROBOTS, ROBOT_DIM), dtype=np.float32)
        for robot in self.robots:
            pos = self.layout.positions[robot.cell]
            target = self.layout.positions[robot.target]
            tokens[robot.rid] = (
                1.0,
                float(robot is me),
                float(robot.rid == 0),
                float(robot.rid == 1),
                float(robot.rid == 2),
                pos[0] / rows,
                pos[1] / cols,
                target[0] / rows,
                target[1] / cols,
                robot.load / self.capacity,
                robot.battery / self.cfg.battery_capacity if self.battery_on else 1.0,
                max(0.0, robot.busy_until - self.time) / 50.0,
            )
        return tokens

    def _battery_floor(self, robot: _Robot) -> float:
        return robot.battery if self.battery_on else math.inf

    def _reachable(self, robot: _Robot, cell: int, battery: float, work_ticks: float = 0.0) -> bool:
        """Hard safety mask: after the move the robot must still be able to reach the charger."""
        if battery == math.inf:
            return True
        cfg = self.cfg
        dist = self.layout.distances
        cost = dist[robot.cell, cell] * cfg.move_drain + work_ticks * cfg.idle_drain
        return battery - cost - dist[cell, self.layout.charger] * cfg.move_drain >= 0

    def _action_kind(self, action: int) -> str:
        if action < self.max_orders:
            return "pick"
        if action < self.charge_action:
            return "deliver"
        return "charge" if action == self.charge_action else "wait"

    # ------------------------------------------------------------------ helpers
    def _pick_layout(self, cfg: EnvConfig, rng: np.random.Generator) -> Layout:
        name = cfg.layouts[int(rng.integers(len(cfg.layouts)))]
        layout = Layout.named(name)
        if cfg.blocked_aisles:
            variant = int(rng.integers(4))
            key = f"{name}:{cfg.blocked_aisles}:{variant}"
            if key not in self._layout_cache:
                self._layout_cache[key] = layout.with_blocked_aisles(np.random.default_rng(variant), cfg.blocked_aisles)
            layout = self._layout_cache[key]
        return layout

    @staticmethod
    def _mean_shelf_hop(layout: Layout) -> float:
        access = np.unique(layout.shelf_access)
        return float(layout.distances[np.ix_(access, access)].mean()) / 2


def _new_stats() -> dict:
    return {
        "orders": 0, "fulfilled": 0, "lost": 0, "deferred": 0, "closed": 0, "on_time": 0,
        "express_orders": 0, "express_on_time": 0, "revenue_total": 0.0, "revenue_captured": 0.0,
        "trucks_missed": 0, "slack_sum": 0.0, "battery_failures": 0, "travel_ticks": 0,
        "deliveries": 0, "items_delivered": 0, "invalid_actions": 0, "decisions": 0,
        "action_counts": {"pick": 0, "deliver": 0, "charge": 0, "wait": 0},
    }  # fmt: skip
