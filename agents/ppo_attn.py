"""MAIN agent: PPO with the permutation-invariant attention encoder and pointer head."""

from __future__ import annotations

from warehouse_env import WarehouseEnv

from .networks import AttentionActorCritic

NET_KIND = "attention"


def build_network(env: WarehouseEnv, d_model: int = 64, n_layers: int = 2, n_heads: int = 4) -> AttentionActorCritic:
    return AttentionActorCritic(
        n_docks=env.dock_count, d_model=d_model, n_layers=n_layers, n_heads=n_heads, use_robots="robots" in env.observation_space.spaces
    )
