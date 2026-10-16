"""Truck departure schedules with reproducible timing jitter."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Truck:
    truck_id: int
    dock: int
    scheduled_departure: float
    actual_departure: float


class TruckScheduler:
    def __init__(self, dock_count: int = 3, interval_range: tuple[int, int] = (120, 180), jitter_std: float = 8.0, seed: int | None = None) -> None:
        if dock_count < 1 or interval_range[0] <= 0 or interval_range[0] > interval_range[1] or jitter_std < 0:
            raise ValueError("invalid truck scheduler configuration")
        self.dock_count = dock_count
        self.interval_range = interval_range
        self.jitter_std = jitter_std
        self.rng = random.Random(seed)

    def generate(self, horizon: int = 720) -> list[Truck]:
        trucks: list[Truck] = []
        truck_id = 0
        for dock in range(self.dock_count):
            departure = self.rng.randint(*self.interval_range)
            while departure <= horizon:
                actual = max(0.0, departure + self.rng.gauss(0.0, self.jitter_std))
                trucks.append(Truck(truck_id, dock, float(departure), actual))
                truck_id += 1
                departure += self.rng.randint(*self.interval_range)
        return sorted(trucks, key=lambda truck: truck.scheduled_departure)