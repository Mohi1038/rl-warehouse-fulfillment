"""Architecture ablation: the same PPO with a flatten + MLP encoder over slack-sorted slots."""

from __future__ import annotations

from warehouse_env import WarehouseEnv

from .networks import MLPActorCritic

NET_KIND = "mlp"


def build_network(env: WarehouseEnv, hidden: int = 256, sort_by_slack: bool = True) -> MLPActorCritic:
    return MLPActorCritic(
        n_docks=env.dock_count,
        n_orders=env.max_orders,
        hidden=hidden,
        use_robots="robots" in env.observation_space.spaces,
        sort_by_slack=sort_by_slack,
    )
