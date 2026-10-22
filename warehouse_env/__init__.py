"""DockRush: deadline-aware warehouse order fulfilment environments."""

from gymnasium.envs.registration import register, registry

from .config import CURRICULUM_LEVELS, EnvConfig, RewardConfig, tiny_config
from .env import WarehouseEnv
from .layout import OOD_LAYOUTS, TRAIN_LAYOUTS, Layout
from .multi_env import MultiRobotWarehouseEnv
from .orders import Order, OrderGenerator
from .tiny_env import TinyWarehouseEnv
from .trucks import Truck, TruckScheduler

for _id, _entry in (
    ("DockRush-v0", "warehouse_env.env:WarehouseEnv"),
    ("DockRushTiny-v0", "warehouse_env.tiny_env:TinyWarehouseEnv"),
    ("DockRushMulti-v0", "warehouse_env.multi_env:MultiRobotWarehouseEnv"),
):
    if _id not in registry:
        register(id=_id, entry_point=_entry)

__all__ = [
    "CURRICULUM_LEVELS",
    "EnvConfig",
    "Layout",
    "MultiRobotWarehouseEnv",
    "OOD_LAYOUTS",
    "Order",
    "OrderGenerator",
    "RewardConfig",
    "TRAIN_LAYOUTS",
    "TinyWarehouseEnv",
    "Truck",
    "TruckScheduler",
    "WarehouseEnv",
    "tiny_config",
]
