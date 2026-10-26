"""Export real simulator episodes and result summaries for the animated website (``site/``).

Every animation on the site replays data written here: robot options with their BFS paths,
order life cycles, truck departures, revenue over time, attention weights and look-ahead
action scores. Writes ``site/public/data/*.json``.

    python experiments/export_site_data.py --seed 100026
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import lookahead  # noqa: E402

import warehouse_env.env as env_module  # noqa: E402
from agents.baselines import make_baseline  # noqa: E402
from agents.common import policy_from_checkpoint  # noqa: E402
from train import eval_env_config  # noqa: E402
from warehouse_env import TRAIN_LAYOUTS, EnvConfig, WarehouseEnv  # noqa: E402
from warehouse_env.env import FULFILLED, LOADED, LOST  # noqa: E402

OUT = ROOT / "site" / "public" / "data"
NOMINAL = EnvConfig(layouts=TRAIN_LAYOUTS)
STATUS = {FULFILLED: "fulfilled", LOST: "lost"}
_generators: list = []


class _RecordingGenerator(env_module.OrderGenerator):
    """Keeps a handle on the episode's order generator so its arrival-rate curve can be exported."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _generators.append(self)


env_module.OrderGenerator = _RecordingGenerator


def bfs_path(layout, start: int, goal: int) -> list[list[int]]:
    src, dst = layout.positions[start], layout.positions[goal]
    prev = {src: None}
    queue = deque([src])
    while queue:
        cell = queue.popleft()
        if cell == dst:
            break
        r, c = cell
        for nxt in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nxt in layout.cell_index and nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)
    path, cell = [], dst
    while cell is not None:
        path.append(list(cell))
        cell = prev[cell]
    return path[::-1]


def layout_json(layout) -> dict:
    pos = layout.positions
    return {
        "name": layout.name,
        "grid": list(layout.grid),
        "docks": [list(pos[d]) for d in layout.docks],
        "charger": list(pos[layout.charger]) if layout.charger is not None else None,
        "packing": list(pos[layout.packing]),
    }


def record(env: WarehouseEnv, seed: int, choose, attention=None, scores=None) -> dict:
    """Play one episode with ``choose(obs, env) -> action`` and record everything the site draws."""
    obs, _ = env.reset(seed=seed)
    layout, cfg = env.layout, env.cfg
    pos = layout.positions
    options, attn, revenue, lookahead_scores = [], [], [[0.0, 0.0]], []
    first_truck: dict[int, int] = {}
    item_events: dict[tuple[int, int], dict] = {}
    done = False
    while not done:
        for order in env.open_orders:
            first_truck.setdefault(order.oid, order.truck.truck_id)
        rid = env.active
        robot = env.robots[rid]
        t0, start_cell, battery = env.time, robot.cell, robot.battery
        if attention is not None:
            weights = attention(obs)
            if weights is not None:
                valid = [(i, o) for i, o in enumerate(env.slots) if o is not None]
                total = sum(float(weights[i]) for i, _ in valid) or 1.0
                attn.append({"t": t0, "w": {str(o.oid): round(float(weights[i]) / total, 4) for i, o in valid}})
        action, extra = choose(obs, env)
        if extra is not None:
            lookahead_scores.append({"t": t0, **extra})
        kind, payload = env._action_kind(action), {}
        if kind == "pick":
            order = env.slots[action]
            item = env._pick_item(robot, order)
            payload = {"order": order.oid, "item": item}
            item_events.setdefault((order.oid, item), {})["claim"] = t0
        elif kind == "deliver":
            dock = action - env.max_orders
            payload = {"dock": dock, "items": [[o.oid, i] for o, i in robot.carried if o.dock == dock]}
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        robot = env.robots[rid]
        end, target = min(robot.busy_until, env.horizon), robot.target
        travel = int(layout.distances[start_cell, target])
        options.append(
            {"r": rid, "k": kind, "t0": round(t0, 2), "t1": round(end, 2), "travel": travel,
             "path": bfs_path(layout, start_cell, target), "bat": round(battery, 1), **payload}
        )  # fmt: skip
        if kind == "pick":
            item_events[(payload["order"], payload["item"])]["pick"] = end
        elif kind == "deliver":
            for oid, item in payload["items"]:
                item_events.setdefault((oid, item), {})["load"] = end
        revenue.append([round(env.time, 2), round(env.stats["revenue_captured"], 3)])

    trucks = [
        {
            "id": t.truck_id,
            "dock": t.dock,
            "scheduled": t.scheduled_departure,
            "actual": round(t.actual_departure, 2),
            "final": t.final,
        }
        for t in env.trucks
    ]
    orders = []
    for o in env.orders:
        items = []
        for i, cell in enumerate(o.access):
            ev = item_events.get((o.oid, i), {})
            loaded = o.item_state[i] == LOADED
            items.append(
                {"cell": list(pos[cell]), "w": o.weights[i], "pick": ev.get("pick"), "load": ev.get("load") if loaded else None}
            )
        status = STATUS.get(o.status, "closed")
        orders.append(
            {"id": o.oid, "arrival": round(o.spec.arrival_time, 2), "tier": o.tier, "vip": o.spec.vip, "value": round(o.spec.value, 2),
             "captured": round(o.value, 2) if o.status == FULFILLED else 0.0, "dock": o.dock, "items": items,
             "truck0": first_truck.get(o.oid, o.truck.truck_id), "truck": o.truck.truck_id, "deferrals": o.deferrals,
             "status": status, "resolved": round(o.truck.actual_departure, 2)}
        )  # fmt: skip
    stats = env.episode_stats()
    out = {
        "seed": seed,
        "horizon": env.horizon,
        "robots": env.n_robots,
        "capacity": env.capacity,
        "battery": cfg.battery_capacity,
        "layout": layout_json(layout),
        "trucks": trucks,
        "orders": orders,
        "options": options,
        "revenue": revenue,
        "revenueTotal": round(env.stats["revenue_total"], 3),
        "stats": {k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in stats.items()},
    }
    if attn:
        out["attention"] = attn
    if lookahead_scores:
        out["lookahead"] = lookahead_scores
    return out


def policy_chooser(policy):
    return lambda obs, env: (policy.act(obs), None)


def lookahead_chooser(policy, samples: int, seed: int):
    rng = np.random.default_rng(seed + 7_000_000)

    def label(env, action):
        kind = env._action_kind(action)
        if kind == "pick":
            return f"pick:{env.slots[action].oid}"
        return f"deliver:{action - env.max_orders}" if kind == "deliver" else kind

    def choose(obs, env):
        actions = np.flatnonzero(obs["action_mask"])
        if len(actions) == 1:
            return int(actions[0]), None
        totals = np.zeros(len(actions))
        for _ in range(samples):
            world = copy.deepcopy(env)
            lookahead.resample_future(world, rng)
            totals += [lookahead.score(world, int(a), policy) for a in actions]
        best = int(np.argmax(totals))
        mean = totals / samples
        return int(actions[best]), {
            "cands": [{"a": label(env, int(a)), "v": round(float(v), 2)} for a, v in zip(actions, mean)],
            "chosen": label(env, int(actions[best])),
            "base": label(env, policy.act(obs)),
        }

    return choose


def curves(name: str) -> list:
    out = []
    for path in sorted((ROOT / "runs" / name).glob("s*/eval.csv")):
        rows = list(csv.DictReader(path.open()))
        out.append([[int(r["update"]) * 4096, round(float(r["task_return"]), 2)] for r in rows])
    return out


def summary() -> dict:
    res = ROOT / "results"
    scen = json.loads((res / "scenarios.json").read_text())

    def mean(runs, key):
        return round(float(np.mean([r["summary"][key] for r in runs])), 2)

    keys = ("revenue_captured_pct", "task_return", "on_time_pct", "express_on_time_pct", "travel_ticks_per_order")
    agents = {
        scenario: {name: {k: mean(runs, k) for k in keys} for name, runs in block["agents"].items() if runs}
        for scenario, block in scen.items()
    }
    la = json.loads((res / "lookahead_ppo_k32.json").read_text())["episodes"]
    look = {
        "with": {k: round(float(np.mean([e[k] for e in la])), 2) for k in keys},
        "without": {k: round(float(np.mean([e["plain"][k] for e in la])), 2) for k in keys},
    }
    multi = json.loads((res / "multi.json").read_text())
    fleet = {
        k: {
            a: round(float(np.mean([r["summary"]["task_return"] for r in runs])), 1)
            for a, v in block.items()
            if (runs := v if isinstance(v, list) else [v]) and isinstance(runs[0], dict) and "summary" in runs[0]
        }
        for k, block in multi.items()
    }
    variants = json.loads((res / "variants.json").read_text())
    return {
        "agents": agents,
        "lookahead": look,
        "fleet": fleet,
        "variants": {
            n: {"validation": round(float(np.mean(v["validation_task_return"])), 1),
                "revenue": mean(v["scenarios"]["nominal"], "revenue_captured_pct"),
                "task_return": mean(v["scenarios"]["nominal"], "task_return")}
            for n, v in variants.items()
        },  # fmt: skip
        "curves": {"ppo_attn": curves("ppo_attn"), "ppo_attn_smdp": curves("ppo_attn_smdp")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=100026)
    parser.add_argument("--samples", type=int, default=32, help="look-ahead sampled futures")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(4)
    ppo, _ = policy_from_checkpoint(ROOT / "models" / "best_ppo_attn.pt")
    episodes = {
        "ppo": lambda: record(WarehouseEnv(NOMINAL), args.seed, policy_chooser(ppo), attention=ppo.attention),
        "greedy": lambda: record(
            WarehouseEnv(NOMINAL), args.seed, policy_chooser(make_baseline("greedy", WarehouseEnv(NOMINAL)))
        ),
        "edf": lambda: record(WarehouseEnv(NOMINAL), args.seed, policy_chooser(make_baseline("edf", WarehouseEnv(NOMINAL)))),
    }
    multi, ckpt = policy_from_checkpoint(ROOT / "models" / "best_multi_k3.pt")
    episodes["fleet"] = lambda: record(WarehouseEnv(eval_env_config(ckpt["config"])), args.seed, policy_chooser(multi))
    episodes["lookahead"] = lambda: record(WarehouseEnv(NOMINAL), args.seed, lookahead_chooser(ppo, args.samples, args.seed))
    for name, run in episodes.items():
        _generators.clear()
        data = run()
        gen = _generators[0]
        data["rate"] = [[t, round(gen.rate(t), 5)] for t in range(0, int(data["horizon"]) + 1, 6)]
        (OUT / f"episode_{name}.json").write_text(json.dumps(data, separators=(",", ":")))
        s = data["stats"]
        print(
            f"  {name:<10} revenue {s['revenue_captured_pct']:.1f}%  orders {s['orders']}  decisions {s['decisions']}", flush=True
        )
    (OUT / "summary.json").write_text(json.dumps(summary(), separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
