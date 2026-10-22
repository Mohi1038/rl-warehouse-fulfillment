"""Configuration objects for the warehouse environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

TIERS = ("express", "standard", "bulk")


@dataclass(frozen=True)
class RewardConfig:
    tier_bonus: tuple[float, float, float] = (1.5, 1.0, 0.7)
    lambda_missed: float = 1.0
    lambda_deferred: float = 0.3
    lambda_cost: float = 0.01
    lambda_battery: float = 50.0
    alpha: float = 0.3


@dataclass(frozen=True)
class EnvConfig:
    layouts: tuple[str, ...] = ("train_a",)
    horizon: int = 720
    max_orders: int = 16
    n_robots: int = 1
    include_robot_tokens: bool = False
    gamma: float = 0.995

    # robot
    capacity: int = 6
    battery_capacity: float = 400.0
    battery_enabled: bool = True
    move_drain: float = 1.0
    idle_drain: float = 0.2
    charge_rate: float = 4.0
    pick_ticks: int = 2
    load_ticks_per_item: int = 1
    wait_ticks: int = 5

    # orders
    base_rate: float = 0.025
    lambda_mult: float = 1.0
    rate_amplitude: float = 0.8
    spike_height: float = 1.5
    spike_width: float = 0.03
    items_range: tuple[int, int] = (1, 3)
    item_weight_range: tuple[int, int] = (1, 3)
    tier_probs: tuple[float, float, float] = (0.2, 0.6, 0.2)
    value_mu: tuple[float, float, float] = (2.7, 2.1, 1.6)
    value_sigma: float = 0.4
    zipf_exponent: float = 1.0
    min_slack: tuple[float, float, float] = (40.0, 60.0, 80.0)
    slack_mult: float = 1.0
    max_total_orders: int | None = None
    shocks_enabled: bool = True
    vip_prob: float = 0.04
    vip_value_mult: float = 5.0
    vip_min_slack: float = 25.0

    # trucks
    truck_interval: tuple[int, int] = (120, 180)
    first_truck_range: tuple[int, int] | None = None
    jitter_std: float = 8.0

    # dynamics extras
    blocked_aisles: int = 0
    domain_randomization: bool = False
    dr_lambda_range: tuple[float, float] = (0.6, 1.4)
    dr_capacity_range: tuple[int, int] = (4, 8)
    dr_jitter_range: tuple[float, float] = (0.0, 12.0)

    reward: RewardConfig = field(default_factory=RewardConfig)

    def replace(self, **changes) -> EnvConfig:
        reward_changes = changes.pop("reward", None)
        config = replace(self, **changes)
        if isinstance(reward_changes, dict):
            config = replace(config, reward=replace(config.reward, **reward_changes))
        elif isinstance(reward_changes, RewardConfig):
            config = replace(config, reward=reward_changes)
        return config

    @classmethod
    def from_dict(cls, data: dict | None) -> EnvConfig:
        data = dict(data or {})
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"unknown env config keys: {sorted(unknown)}")
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = tuple(value)
        reward = data.pop("reward", None)
        config = cls(**data)
        return config.replace(reward=reward) if reward else config

    def to_dict(self) -> dict:
        return asdict(self)


# Curriculum levels from PLAN.md §5.1. Level 2 is the nominal evaluation setting.
CURRICULUM_LEVELS: tuple[dict, ...] = (
    {"lambda_mult": 0.4, "slack_mult": 2.0, "battery_enabled": False, "shocks_enabled": False, "blocked_aisles": 0},
    {"lambda_mult": 0.7, "slack_mult": 1.5, "battery_enabled": True, "shocks_enabled": False, "blocked_aisles": 0},
    {"lambda_mult": 1.0, "slack_mult": 1.0, "battery_enabled": True, "shocks_enabled": True, "blocked_aisles": 0},
    {"lambda_mult": 1.3, "slack_mult": 0.8, "battery_enabled": True, "shocks_enabled": True, "blocked_aisles": 3},
)


def apply_level(config: EnvConfig, level: int) -> EnvConfig:
    """Compose a curriculum level with ``config``: rates and slack multiply, features can only be switched off."""
    spec = CURRICULUM_LEVELS[level]
    return config.replace(
        lambda_mult=config.lambda_mult * spec["lambda_mult"],
        slack_mult=config.slack_mult * spec["slack_mult"],
        battery_enabled=config.battery_enabled and spec["battery_enabled"],
        shocks_enabled=config.shocks_enabled and spec["shocks_enabled"],
        blocked_aisles=max(config.blocked_aisles, spec["blocked_aisles"]),
    )


def tiny_config(**overrides) -> EnvConfig:
    """7x5 grid, at most 3 orders, capacity 2, one dock, no battery (PLAN.md §4.4).

    All orders are revealed at the start of the shift and values are set by tier, so
    the tabular state is exact and the hindsight optimum is attainable by a policy.
    """
    base = EnvConfig(
        layouts=("tiny",),
        horizon=60,
        max_orders=3,
        capacity=2,
        battery_enabled=False,
        base_rate=20.0,
        rate_amplitude=0.0,
        value_sigma=0.0,
        spike_height=0.0,
        items_range=(1, 2),
        item_weight_range=(1, 1),
        min_slack=(10.0, 12.0, 15.0),
        max_total_orders=3,
        shocks_enabled=False,
        truck_interval=(30, 30),
        first_truck_range=(30, 30),
        jitter_std=0.0,
        gamma=0.999,
    )
    return base.replace(**overrides)
