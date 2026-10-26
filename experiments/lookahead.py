"""Decision-time look-ahead on top of a base policy, without knowledge of the future.

At every decision the planner scores each valid action by simulating it on ``K`` sampled
futures and finishing the shift with the base policy (Greedy or a PPO checkpoint). A sampled
future keeps everything the agent could know (orders already arrived, robot state, the truck
schedule) and redraws what it cannot: orders that have not arrived yet and the hidden
departure jitter of trucks that have not left. All actions are scored on the same ``K``
futures. The action with the highest mean revenue captured is executed in the real episode.

``--clairvoyant`` scores actions on the true future instead (one rollout per action); it is
a reference for how much of the gain comes from knowing the future.

    python experiments/lookahead.py --base greedy --samples 8 --episodes 20
    python experiments/lookahead.py --base ppo --samples 8 --episodes 20
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.baselines import make_baseline  # noqa: E402
from agents.common import EVAL_SEEDS, policy_from_checkpoint  # noqa: E402
from warehouse_env import TRAIN_LAYOUTS, EnvConfig, WarehouseEnv  # noqa: E402
from warehouse_env.orders import OrderGenerator  # noqa: E402

NOMINAL = EnvConfig(layouts=TRAIN_LAYOUTS)
MODEL = ROOT / "models" / "best_ppo_attn.pt"
_BASE = None  # per-process base policy


def resample_future(env: WarehouseEnv, rng: np.random.Generator) -> None:
    """Replace, in place, every part of ``env`` the agent cannot observe with a fresh sample."""
    cfg, now = env.cfg, env.time
    future = OrderGenerator(
        len(env.layout.shelves),
        env.dock_count,
        base_rate=cfg.base_rate * cfg.lambda_mult,
        horizon=cfg.horizon,
        seed=rng,
        rate_amplitude=cfg.rate_amplitude,
        spike_height=cfg.spike_height,
        spike_width=cfg.spike_width,
        items_range=cfg.items_range,
        item_weight_range=cfg.item_weight_range,
        tier_probs=cfg.tier_probs,
        value_mu=cfg.value_mu,
        value_sigma=cfg.value_sigma,
        zipf_exponent=cfg.zipf_exponent,
        vip_prob=cfg.vip_prob if cfg.shocks_enabled else 0.0,
        vip_value_mult=cfg.vip_value_mult,
    ).generate()
    past = env.specs[: env._next_spec]
    env.specs = past + [replace(s, order_id=len(past) + i) for i, s in enumerate(o for o in future if o.arrival_time > now)]
    sigma = cfg.jitter_std
    for truck in env.trucks[env._next_truck :]:
        if not truck.final:
            jitter = float(np.clip(rng.normal(0.0, sigma), -3 * sigma, 3 * sigma))
            object.__setattr__(truck, "actual_departure", max(now + 1e-6, 1.0, truck.scheduled_departure + jitter))
    env.trucks[env._next_truck :] = sorted(env.trucks[env._next_truck :], key=lambda t: (t.actual_departure, t.dock))


def finish(env: WarehouseEnv, obs, policy) -> float:
    done = env.time >= env.horizon
    while not done:
        obs, _, terminated, truncated, _ = env.step(policy.act(obs))
        done = terminated or truncated
    return env.stats["revenue_captured"]


def score(env: WarehouseEnv, action: int, policy) -> float:
    env = copy.deepcopy(env)
    obs, _, terminated, truncated, _ = env.step(action)
    return env.stats["revenue_captured"] if terminated or truncated else finish(env, obs, policy)


def run_episode(seed: int, base: str, samples: int, clairvoyant: bool) -> dict:
    policy = make_base(base)
    env = WarehouseEnv(NOMINAL)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7_000_000)
    done, info = False, {}
    while not done:
        actions = np.flatnonzero(obs["action_mask"])
        if len(actions) == 1:
            action = int(actions[0])
        elif clairvoyant:
            action = int(max(actions, key=lambda a: score(env, int(a), policy)))
        else:
            totals = np.zeros(len(actions))
            for _ in range(samples):
                world = copy.deepcopy(env)
                resample_future(world, rng)
                totals += [score(world, int(a), policy) for a in actions]
            action = int(actions[np.argmax(totals)])
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return {"seed": seed, **info["episode_stats"]}


def make_base(base: str):
    global _BASE
    if _BASE is None:
        if base == "ppo":
            import torch

            torch.set_num_threads(1)
            _BASE = policy_from_checkpoint(MODEL)[0]
        else:
            _BASE = make_baseline(base, WarehouseEnv(NOMINAL))
    return _BASE


def plain_episode(seed: int, base: str) -> dict:
    env = WarehouseEnv(NOMINAL)
    obs, _ = env.reset(seed=seed)
    finish(env, obs, make_base(base))
    return {"seed": seed, **env.episode_stats()}


def job(args):
    seed, base, samples, clairvoyant = args
    start = time.time()
    return {**run_episode(seed, base, samples, clairvoyant), "seconds": time.time() - start, "plain": plain_episode(seed, base)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", choices=("greedy", "edf", "ppo"), default="greedy")
    parser.add_argument("--samples", type=int, default=8, help="sampled futures per decision")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--clairvoyant", action="store_true")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out")
    args = parser.parse_args()
    seeds = EVAL_SEEDS[: args.episodes]
    with Pool(args.jobs) as pool:
        episodes = pool.map(job, [(int(s), args.base, args.samples, args.clairvoyant) for s in seeds])
    keys = ("revenue_captured_pct", "task_return", "on_time_pct", "express_on_time_pct")
    tag = f"{args.base} + {'clairvoyant' if args.clairvoyant else f'look-ahead K={args.samples}'}"
    for name, rows in ((args.base, [e["plain"] for e in episodes]), (tag, episodes)):
        print(f"{name:<28}" + "  ".join(f"{k[:12]}={np.mean([r[k] for r in rows]):6.1f}" for k in keys))
    print(
        f"seconds per episode: mean {np.mean([e['seconds'] for e in episodes]):.0f}, max {max(e['seconds'] for e in episodes):.0f}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "episodes": episodes}, default=float))


if __name__ == "__main__":
    main()
