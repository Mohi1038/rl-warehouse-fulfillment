"""Multi-robot variant (PLAN.md §7).

Decisions are asynchronous: the event queue hands control to whichever robot frees
up next, and one shared policy acts for all robots. The observation is ego-centric
(``global`` describes the acting robot) plus ``robots`` tokens (position, target,
load, battery, robot-ID one-hot) for every robot. An order picked by one robot is
masked for the others until it is fully picked, which prevents duplicate work.
"""

from __future__ import annotations

from .config import EnvConfig
from .env import WarehouseEnv


class MultiRobotWarehouseEnv(WarehouseEnv):
    def __init__(self, config: EnvConfig | dict | None = None, n_robots: int = 2, render_mode: str | None = None, **overrides) -> None:
        if isinstance(config, EnvConfig):
            config = config.replace(n_robots=n_robots, include_robot_tokens=True, **overrides)
        else:
            config = EnvConfig.from_dict({**(config or {}), **overrides, "n_robots": n_robots, "include_robot_tokens": True})
        super().__init__(config, render_mode=render_mode)
