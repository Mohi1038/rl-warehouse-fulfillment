"""Reward decomposition for warehouse transitions (PLAN.md §2.8)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RewardConfig

TERM_NAMES = ("fulfilled", "lost", "deferred", "operating_cost", "battery", "shaping")


@dataclass
class RewardAccumulator:
    """Collects raw order outcomes during one option and turns them into reward terms."""

    fulfilled: float = 0.0
    lost: float = 0.0
    deferred: float = 0.0

    def terms(self, config: RewardConfig, tau: float, robots: int, stranded: bool, shaping: float) -> dict[str, float]:
        return {
            "fulfilled": self.fulfilled,
            "lost": -config.lambda_missed * self.lost,
            "deferred": -config.lambda_deferred * 0.5 * self.deferred,
            "operating_cost": -config.lambda_cost * tau * robots,
            "battery": -config.lambda_battery if stranded else 0.0,
            "shaping": shaping,
        }


def total(terms: dict[str, float]) -> float:
    """Use the builtin ``sum`` (compensated on 3.12+) so ``sum(info["reward_terms"].values()) == reward`` exactly."""
    return sum(terms[name] for name in TERM_NAMES)


def zero_terms() -> dict[str, float]:
    return dict.fromkeys(TERM_NAMES, 0.0)
