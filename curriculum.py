"""Curriculum over the difficulty levels of PLAN.md §5.1.

Promotes when the on-time rate over the last ``window`` episodes exceeds ``threshold``.
A fallback promotes after ``max_updates_per_level`` updates so a run cannot stall on a
level it will never master, which would leave the nominal task untrained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    enabled: bool = True
    start_level: int = 0
    max_level: int = 2
    threshold: float = 80.0
    window: int = 50
    max_updates_per_level: int | None = 150


class Curriculum:
    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        self.level = config.start_level if config.enabled else None
        self.history: deque[float] = deque(maxlen=config.window)
        self.updates_at_level = 0

    def record(self, on_time_pct: float) -> None:
        if self.level is not None:
            self.history.append(on_time_pct)

    def step(self) -> bool:
        """Call once per update. Returns True when the level changed."""
        if self.level is None or self.level >= self.config.max_level:
            return False
        self.updates_at_level += 1
        full = len(self.history) == self.history.maxlen
        mastered = full and sum(self.history) / len(self.history) > self.config.threshold
        timed_out = self.config.max_updates_per_level is not None and self.updates_at_level >= self.config.max_updates_per_level
        if mastered or timed_out:
            self.level += 1
            self.history.clear()
            self.updates_at_level = 0
            return True
        return False

    @property
    def on_time(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0
