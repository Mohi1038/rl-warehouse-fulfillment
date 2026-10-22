"""Evaluate trained agents and baselines on held-out seeds.

    python evaluate.py --model models/best_ppo_attn.pt --episodes 100 --render
    python evaluate.py --agents ppo_attn=models/best_ppo_attn.pt edf greedy fifo random --episodes 100
    python evaluate.py --agents ppo_attn=models/best_ppo_attn.pt edf --set layouts=[ood_a,ood_b] --out results/ood.json

Agents are ``name=path.pt`` checkpoints or baseline names (random, fifo, edf, greedy,
edf_charge). The environment is the nominal evaluation setting of the first checkpoint
(or the default nominal config when only baselines are given), with ``--set`` overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from agents.baselines import BASELINES, make_baseline
from agents.common import EVAL_SEEDS, baseline_act_batch, evaluate_policy, policy_from_checkpoint, summarize
from train import eval_env_config
from warehouse_env import TRAIN_LAYOUTS, EnvConfig, WarehouseEnv

TABLE_KEYS = (
    "return",
    "revenue_captured_pct",
    "on_time_pct",
    "express_on_time_pct",
    "trucks_missed",
    "mean_slack_at_delivery",
    "battery_failures",
    "travel_ticks_per_order",
    "items_per_trip",
)

NOMINAL = EnvConfig(layouts=TRAIN_LAYOUTS)


def parse_agents(specs: list[str]) -> list[tuple[str, str | None]]:
    agents = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
            agents.append((name, path))
        elif spec in BASELINES:
            agents.append((spec, None))
        elif spec.endswith(".pt"):
            agents.append((Path(spec).stem.removeprefix("best_"), spec))
        else:
            raise SystemExit(f"unknown agent {spec!r}: use name=path.pt or one of {sorted(BASELINES)}")
    return agents


def resolve_env(agents, overrides: list[str]) -> EnvConfig:
    config = NOMINAL
    for _, path in agents:
        if path:
            _, ckpt = policy_from_checkpoint(path)
            config = eval_env_config(ckpt["config"])
            break
    changes = {}
    for item in overrides:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        changes[key] = tuple(value) if isinstance(value, list) else value
    return config.replace(**changes) if changes else config


def make_actor(name: str, path: str | None, config: EnvConfig, stochastic: bool = False):
    if path is None:
        policy = make_baseline(name, WarehouseEnv(config))
        return baseline_act_batch(policy), policy.act, None
    policy, _ = policy_from_checkpoint(path, deterministic=not stochastic)
    attention = policy.attention if policy.kind == "actor_critic" and hasattr(policy.net, "encoder") else None
    return policy.act_batch, policy.act, attention


def evaluate_agents(agents, config: EnvConfig, seeds, level=None, stochastic=False) -> dict[str, list[dict]]:
    results = {}
    for name, path in agents:
        act_batch, _, _ = make_actor(name, path, config, stochastic)
        results[name] = evaluate_policy(act_batch, config, seeds, level=level)
    return results


def print_table(results: dict[str, list[dict]]) -> None:
    header = f"{'agent':<14}" + "".join(f"{k[:14]:>16}" for k in TABLE_KEYS)
    print(header)
    print("-" * len(header))
    for name, episodes in results.items():
        cells = []
        for key in TABLE_KEYS:
            values = np.array([e[key] for e in episodes], dtype=float)
            ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            cells.append(f"{values.mean():>9.2f}±{ci:<6.2f}")
        print(f"{name:<14}" + "".join(cells))


def render_gif(name: str, path: str | None, config: EnvConfig, out: Path, seed: int) -> Path:
    from warehouse_env.render import record_gif

    _, act, attention = make_actor(name, path, config)
    env = WarehouseEnv(config, render_mode="rgb_array")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = record_gif(env, act, str(out), seed=seed, attention_fn=attention, max_frames=400)
    env.close()
    print(f"saved {out} ({frames} frames)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="checkpoint to evaluate (shorthand for --agents name=path)")
    parser.add_argument("--agents", nargs="*", default=[])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--set", nargs="*", default=[], help="env overrides, e.g. lambda_mult=1.5 capacity=4")
    parser.add_argument("--level", type=int, default=None, help="evaluate at a curriculum level instead of nominal")
    parser.add_argument("--stochastic", action="store_true", help="sample from the policy instead of argmax")
    parser.add_argument("--render", action="store_true", help="record a GIF of the first agent on the first seed")
    parser.add_argument("--gif", default=None, help="GIF path (default assets/<agent>.gif)")
    parser.add_argument("--out", help="write per-episode results as JSON")
    args = parser.parse_args()

    specs = ([f"{Path(args.model).stem.removeprefix('best_')}={args.model}"] if args.model else []) + args.agents
    if not specs:
        specs = ["random", "fifo", "edf", "greedy", "edf_charge"]
    agents = parse_agents(specs)
    config = resolve_env(agents, args.set)
    seeds = EVAL_SEEDS[: args.episodes]
    results = evaluate_agents(agents, config, seeds, args.level, args.stochastic)
    print_table(results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"env": config.to_dict(), "seeds": list(seeds), "summary": {k: summarize(v) for k, v in results.items()}, "episodes": results}
        Path(args.out).write_text(json.dumps(payload, indent=1, default=float))
        print(f"wrote {args.out}")
    if args.render:
        name, path = agents[0]
        render_gif(name, path, config, Path(args.gif or f"assets/{name}.gif"), int(seeds[0]))


if __name__ == "__main__":
    main()
