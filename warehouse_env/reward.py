"""Reward decomposition for warehouse transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardTerms:
    fulfilled: float = 0.0
    lost: float = 0.0
    deferred: float = 0.0
    operating_cost: float = 0.0
    battery: float = 0.0
    shaping: float = 0.0

    @property
    def total(self) -> float:
        return sum((self.fulfilled, self.lost, self.deferred, self.operating_cost, self.battery, self.shaping))

    def as_dict(self) -> dict[str, float]:
        return {
            "fulfilled": self.fulfilled,
            "lost": self.lost,
            "deferred": self.deferred,
            "operating_cost": self.operating_cost,
            "battery": self.battery,
            "shaping": self.shaping,
        }