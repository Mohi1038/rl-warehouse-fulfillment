"""Sanity check: tabular SMDP Q-learning on the tiny environment (PLAN.md §4.4).

State = (robot cell, time bucket, per-order (slot, tier, unpicked, carried, loaded,
deadline)), from ``WarehouseEnv.tabular_state``. The update discounts by ``γ^τ``.
Convergence is measured against a clairvoyant hindsight-optimal upper bound and the
heuristics; matching PPO on the same instances validates env + reward.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from warehouse_env import EnvConfig, WarehouseEnv
from warehouse_env.tiny_env import hindsight_optimal_return

from .baselines import make_baseline
from .common import CSVLogger, baseline_act_batch, evaluate_policy

TINY_EVAL_SEEDS = tuple(range(100_000, 100_200))


@dataclass
class QLearningConfig:
    episodes: int = 60_000
    alpha: float = 0.1
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_fraction: float = 0.6
    eval_every: int = 2_000
    eval_episodes: int = 200
    oracle_episodes: int = 200


class TabularQ:
    def __init__(self, n_actions: int) -> None:
        self.q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

    def greedy(self, state: tuple, mask: np.ndarray) -> int:
        values = np.where(mask.astype(bool), self.q[state], -np.inf)
        return int(np.argmax(values))

    def value(self, state: tuple, mask: np.ndarray) -> float:
        return float(np.max(np.where(mask.astype(bool), self.q[state], -np.inf)))


def evaluate_q(agent: TabularQ, config: EnvConfig, seeds) -> float:
    env = WarehouseEnv(config)
    returns = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.greedy(env.tabular_state(), obs["action_mask"]))
            done = terminated or truncated
        returns.append(info["episode_stats"]["task_return"])
    return float(np.mean(returns))


def oracle_returns(config: EnvConfig, seeds) -> list[float]:
    env = WarehouseEnv(config)
    out = []
    for seed in seeds:
        env.reset(seed=int(seed))
        out.append(hindsight_optimal_return(env))
    return out


def train_q_learning(config: EnvConfig, cfg: QLearningConfig, seed: int, run_dir: Path) -> dict:
    env = WarehouseEnv(config)
    agent = TabularQ(env.action_space.n)
    rng = np.random.default_rng(seed)
    gamma = config.gamma
    seeds = TINY_EVAL_SEEDS[: cfg.eval_episodes]
    log = CSVLogger(run_dir / "eval.csv")
    curve = []
    for episode in range(1, cfg.episodes + 1):
        eps = cfg.eps_start + min(1.0, episode / (cfg.eps_fraction * cfg.episodes)) * (cfg.eps_end - cfg.eps_start)
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        state = env.tabular_state()
        done = False
        while not done:
            mask = obs["action_mask"]
            action = int(rng.choice(np.flatnonzero(mask))) if rng.random() < eps else agent.greedy(state, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.tabular_state()
            target = reward + (0.0 if done else gamma ** info["tau"] * agent.value(next_state, obs["action_mask"]))
            agent.q[state][action] += cfg.alpha * (target - agent.q[state][action])
            state = next_state
        if episode % cfg.eval_every == 0 or episode == cfg.episodes:
            score = evaluate_q(agent, config, seeds)
            curve.append((episode, score))
            log.log({"episode": episode, "task_return": score, "states": len(agent.q), "epsilon": eps})
            print(f"  episode {episode}: greedy task return {score:.2f} ({len(agent.q)} states)", flush=True)

    oracle = oracle_returns(config, seeds[: cfg.oracle_episodes])
    baselines = {}
    for name in ("random", "fifo", "edf", "greedy"):
        results = evaluate_policy(baseline_act_batch(make_baseline(name, env)), config, seeds)
        baselines[name] = float(np.mean([r["task_return"] for r in results]))
    result = {
        "q_learning": curve[-1][1],
        "hindsight_optimal": float(np.mean(oracle)),
        "q_fraction_of_optimal": curve[-1][1] / float(np.mean(oracle)),
        "baselines": baselines,
        "states": len(agent.q),
        "curve": curve,
    }
    np.save(run_dir / "q_table_size.npy", np.array([len(agent.q)]))
    (run_dir / "q_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    return result
