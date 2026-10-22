"""Truck departure schedules with reproducible timing jitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Truck:
    truck_id: int
    dock: int
    scheduled_departure: float
    actual_departure: float
    final: bool = False


class TruckScheduler:
    """Per-dock departures every ``interval_range`` ticks plus an end-of-shift truck at ``T``.

    The agent only observes ``scheduled_departure``; the truck actually leaves at
    ``scheduled + N(0, σ)`` clipped to ±3σ, so a safety margin has to be learned.
    The final truck of every dock leaves exactly at the horizon so that every order
    is resolved by the end of the episode.
    """

    def __init__(
        self,
        dock_count: int = 3,
        interval_range: tuple[int, int] = (120, 180),
        jitter_std: float = 8.0,
        seed: int | np.random.Generator | None = None,
        first_range: tuple[int, int] | None = None,
    ) -> None:
        if dock_count < 1 or interval_range[0] <= 0 or interval_range[0] > interval_range[1] or jitter_std < 0:
            raise ValueError("invalid truck scheduler configuration")
        self.dock_count = dock_count
        self.interval_range = interval_range
        self.first_range = first_range or interval_range
        self.jitter_std = jitter_std
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def generate(self, horizon: int = 720) -> list[Truck]:
        trucks: list[Truck] = []
        margin = 3 * self.jitter_std + 1
        for dock in range(self.dock_count):
            departure = int(self.rng.integers(self.first_range[0], self.first_range[1] + 1))
            while departure < horizon - margin:
                jitter = float(np.clip(self.rng.normal(0.0, self.jitter_std), -3 * self.jitter_std, 3 * self.jitter_std))
                trucks.append(Truck(-1, dock, float(departure), max(1.0, departure + jitter)))
                departure += int(self.rng.integers(self.interval_range[0], self.interval_range[1] + 1))
            trucks.append(Truck(-1, dock, float(horizon), float(horizon), final=True))
        trucks.sort(key=lambda truck: (truck.actual_departure, truck.dock))
        return [Truck(index, t.dock, t.scheduled_departure, t.actual_departure, t.final) for index, t in enumerate(trucks)]
