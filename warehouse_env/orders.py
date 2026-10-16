"""Stochastic order arrivals for the warehouse environment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    order_id: int
    arrival_time: float
    items: tuple[int, ...]
    tier: str
    value: float
    weight: int
    dock: int
    scheduled_departure: float


class OrderGenerator:
    """Generate orders with a sinusoidally varying Poisson arrival rate."""

    def __init__(self, shelf_count: int, dock_count: int = 3, base_rate: float = 0.04, horizon: int = 720, seed: int | None = None) -> None:
        if shelf_count < 1 or dock_count < 1 or base_rate < 0:
            raise ValueError("shelf_count and dock_count must be positive; base_rate cannot be negative")
        self.shelf_count = shelf_count
        self.dock_count = dock_count
        self.base_rate = base_rate
        self.horizon = horizon
        self.rng = random.Random(seed)

    def rate(self, time: float) -> float:
        phase = 2 * math.pi * time / self.horizon - math.pi / 2
        return self.base_rate * (1.0 + 0.8 * math.sin(phase))

    def generate(self) -> list[Order]:
        arrivals: list[Order] = []
        time = 0.0
        order_id = 0
        max_rate = self.base_rate * 1.8
        if max_rate == 0:
            return arrivals
        while time < self.horizon:
            time += self.rng.expovariate(max_rate)
            if time >= self.horizon or self.rng.random() > self.rate(time) / max_rate:
                continue
            item_count = self.rng.randint(1, 4)
            items = tuple(self.rng.randrange(self.shelf_count) for _ in range(item_count))
            tier = self.rng.choices(("express", "standard", "bulk"), weights=(20, 60, 20), k=1)[0]
            value = self.rng.lognormvariate({"express": 3.0, "standard": 2.0, "bulk": 1.0}[tier], 0.35)
            weight = sum(self.rng.randint(1, 3) for _ in items)
            dock = self.rng.randrange(self.dock_count)
            departure = math.ceil((time + 60) / 150) * 150
            arrivals.append(Order(order_id, time, items, tier, value, weight, dock, departure))
            order_id += 1
        return arrivals