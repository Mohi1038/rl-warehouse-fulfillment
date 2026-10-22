"""Evaluate every trained run and baseline on the held-out seeds; writes ``results/*.json``.

E1/E3/E5  scenarios.json      all algorithms x seeds on nominal, OOD layouts and stress scenarios
E4        generalisation.json λ x capacity grid, PPO-attention (with / without randomisation) vs EDF
E6        scaling.json        zero-shot 24/32 order slots for the attention policy
E7        ablations.json      every ablation x seed on the nominal scenario
E8        behaviour.json      attention vs slack/value, actions by time of day, items per trip
M11       multi.json          1 vs 2 vs 3 robots, per-robot pick zones
M5        tiny.json           tabular Q vs PPO vs hindsight optimum on the tiny env
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.baselines import make_baseline  # noqa: E402
from agents.common import EVAL_SEEDS, baseline_act_batch, evaluate_policy, policy_from_checkpoint, summarize  # noqa: E402
from agents.q_learning import TINY_EVAL_SEEDS  # noqa: E402
from train import eval_env_config  # noqa: E402
from warehouse_env import TRAIN_LAYOUTS, EnvConfig, WarehouseEnv  # noqa: E402

BASELINE_NAMES = ("random", "fifo", "edf", "greedy", "edf_charge")
NOMINAL = EnvConfig(layouts=TRAIN_LAYOUTS)
SCENARIOS = {
    "nominal": {},
    "ood_a": {"layouts": ("ood_a",)},
    "ood_b": {"layouts": ("ood_b",)},
    "rush_x1.5": {"lambda_mult": 1.5},
    "capacity_4": {"capacity": 4},
}
KEEP = ("return", "task_return", "revenue_captured_pct", "on_time_pct", "express_on_time_pct", "trucks_missed",
        "mean_slack_at_delivery", "battery_failures", "travel_ticks_per_order", "items_per_trip")  # fmt: skip


def seed_dirs(runs: Path, name: str) -> list[Path]:
    return sorted((p for p in (runs / name).glob("s*") if (p / "best.pt").exists()), key=lambda p: int(p.name[1:]))


def compact(episodes: list[dict]) -> dict:
    return {
        "summary": {k: v for k, v in summarize(episodes).items() if k in KEEP},
        "task_return": [e["task_return"] for e in episodes],
    }


def eval_model(path: Path, config: EnvConfig, seeds) -> list[dict]:
    policy, _ = policy_from_checkpoint(path)
    return evaluate_policy(policy.act_batch, config, seeds)


def eval_baseline(name: str, config: EnvConfig, seeds) -> list[dict]:
    return evaluate_policy(baseline_act_batch(make_baseline(name, WarehouseEnv(config))), config, seeds)


def scenarios(runs: Path, seeds) -> dict:
    out: dict = {}
    for scenario, changes in SCENARIOS.items():
        config = NOMINAL.replace(**changes)
        out[scenario] = {"agents": {}}
        for algo in ("ppo_attn", "ppo_mlp", "dqn"):
            out[scenario]["agents"][algo] = [compact(eval_model(d / "best.pt", config, seeds)) for d in seed_dirs(runs, algo)]
        for name in BASELINE_NAMES:
            out[scenario]["agents"][name] = [compact(eval_baseline(name, config, seeds))]
        print(f"  scenario {scenario} done", flush=True)
    return out


def generalisation(runs: Path, episodes: int) -> dict:
    lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    capacities = [3, 4, 5, 6, 7, 8, 9, 10]
    seeds = EVAL_SEEDS[:episodes]
    agents = {name: seed_dirs(runs, name) for name in ("ppo_attn", "abl_no_dr")}
    grid: dict = {name: np.zeros((len(lambdas), len(capacities))).tolist() for name in (*agents, "edf", "greedy")}
    for i, lam in enumerate(lambdas):
        for j, cap in enumerate(capacities):
            config = NOMINAL.replace(lambda_mult=lam, capacity=cap)
            for name, dirs in agents.items():
                if dirs:
                    grid[name][i][j] = float(
                        np.mean([summarize(eval_model(d / "best.pt", config, seeds))["task_return"] for d in dirs])
                    )
            for name in ("edf", "greedy"):
                grid[name][i][j] = summarize(eval_baseline(name, config, seeds))["task_return"]
    return {"lambda_mult": lambdas, "capacity": capacities, "task_return": grid}


def scaling(runs: Path, seeds) -> dict:
    out: dict = defaultdict(dict)
    for lam in (1.0, 2.0):
        for slots in (16, 24, 32):
            config = NOMINAL.replace(max_orders=slots, lambda_mult=lam)
            key = f"lambda_{lam}"
            out[key][slots] = {
                "ppo_attn": [
                    summarize(eval_model(d / "best.pt", config, seeds))["task_return"] for d in seed_dirs(runs, "ppo_attn")
                ],
                "edf": summarize(eval_baseline("edf", config, seeds))["task_return"],
                "greedy": summarize(eval_baseline("greedy", config, seeds))["task_return"],
            }
            if slots == 16:
                out[key][slots]["ppo_mlp"] = [
                    summarize(eval_model(d / "best.pt", config, seeds))["task_return"] for d in seed_dirs(runs, "ppo_mlp")
                ]
    return out


def ablations(runs: Path, seeds) -> dict:
    variants = {"full": "ppo_attn", "no_attention": "ppo_mlp"}
    variants.update({p.name.removeprefix("abl_"): p.name for p in sorted(runs.glob("abl_*"))})
    out = {}
    for label, name in variants.items():
        dirs = seed_dirs(runs, name)[:3]
        if not dirs:
            continue
        config = eval_env_config(policy_from_checkpoint(dirs[0] / "best.pt")[1]["config"])
        out[label] = [summarize(eval_model(d / "best.pt", config, seeds))["task_return"] for d in dirs]
        out[label + "__env"] = {"battery_enabled": config.battery_enabled}
    return out


def behaviour(runs: Path, seeds) -> dict:
    model = ROOT / "models" / "best_ppo_attn.pt"
    policy, _ = policy_from_checkpoint(model)
    pairs: list[tuple] = []

    def act_with_attention(obs):
        actions = policy.act_batch(obs, need_weights=True)
        for row, action in enumerate(actions):
            valid = obs["orders"][row, :, 0] > 0.5
            if valid.sum() < 2:
                continue
            for slot in np.flatnonzero(valid):
                features = obs["orders"][row, slot]
                pairs.append(
                    (
                        float(policy.last_attention[row, slot]),
                        float(features[8]),
                        float(features[1]) * 50,
                        int(np.argmax(features[2:5])),
                        int(action == slot),
                        int(valid.sum()),
                    )
                )
        return actions

    timeline: dict = {}
    trips: dict = {}

    def recorder(name):
        tod = np.zeros((12, 4))
        sizes: list[int] = []
        kinds = {"pick": 0, "deliver": 1, "charge": 2, "wait": 3}

        def on_step(i, env, obs, action, reward, info):
            tod[min(11, int(obs["global"][0] * 12)), kinds[env._action_kind(action)]] += 1
            sizes.extend(int(e.split(":")[2]) for e in info["events"] if e.startswith("deliver:"))

        return on_step, tod, sizes

    for name in ("ppo_attn", "edf", "greedy"):
        on_step, tod, sizes = recorder(name)
        act = act_with_attention if name == "ppo_attn" else baseline_act_batch(make_baseline(name, WarehouseEnv(NOMINAL)))
        evaluate_policy(act, NOMINAL, seeds, on_step=on_step)
        timeline[name] = tod.tolist()
        trips[name] = np.bincount(np.array(sizes, dtype=int), minlength=7).tolist()
    return {"attention_pairs": pairs[:30000], "pair_fields": ["attention", "slack", "value", "tier", "chosen", "n_valid"],
            "actions_by_time": timeline, "action_kinds": ["pick", "deliver", "charge", "wait"], "items_per_trip": trips}  # fmt: skip


def multi(runs: Path, seeds) -> dict:
    out = {}
    for k in (1, 2, 3):
        dirs = seed_dirs(runs, f"multi_k{k}")
        if not dirs:
            continue
        config = eval_env_config(policy_from_checkpoint(dirs[0] / "best.pt")[1]["config"])
        columns = np.zeros((k, WarehouseEnv(config).layout.shape[1]))

        def on_step(i, env, obs, action, reward, info, columns=columns):
            if action < env.max_orders:
                robot = int(np.argmax(obs["robots"][:, 1]))
                columns[robot, env.layout.positions[env.robots[robot].target][1]] += 1

        best = ROOT / "models" / f"best_multi_k{k}.pt"
        policy, _ = policy_from_checkpoint(best if best.exists() else dirs[0] / "best.pt")
        evaluate_policy(policy.act_batch, config, seeds, on_step=on_step)
        out[f"k{k}"] = {
            "ppo_attn": [compact(eval_model(d / "best.pt", config, seeds)) for d in dirs],
            "edf": compact(eval_baseline("edf", config, seeds)),
            "greedy": compact(eval_baseline("greedy", config, seeds)),
            "pick_columns": columns.tolist(),
        }
    return out


def tiny(runs: Path) -> dict:
    summaries = [json.loads(p.read_text()) for p in sorted(runs.glob("tiny_q/s*/q_summary.json"))]
    if not summaries:
        return {}
    config = (
        eval_env_config(policy_from_checkpoint(seed_dirs(runs, "tiny_ppo")[0] / "best.pt")[1]["config"])
        if seed_dirs(runs, "tiny_ppo")
        else None
    )
    ppo = (
        [summarize(eval_model(d / "best.pt", config, TINY_EVAL_SEEDS))["task_return"] for d in seed_dirs(runs, "tiny_ppo")]
        if config
        else []
    )
    return {
        "q_learning": [s["q_learning"] for s in summaries],
        "q_curves": [s["curve"] for s in summaries],
        "hindsight_optimal": summaries[0]["hindsight_optimal"],
        "baselines": summaries[0]["baselines"],
        "ppo_attn": ppo,
        "states": [s["states"] for s in summaries],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    runs = ROOT / args.runs
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    seeds = EVAL_SEEDS[: args.episodes]
    jobs = {
        "scenarios": lambda: scenarios(runs, seeds),
        "generalisation": lambda: generalisation(runs, max(5, args.episodes // 3)),
        "scaling": lambda: scaling(runs, seeds),
        "ablations": lambda: ablations(runs, seeds),
        "behaviour": lambda: behaviour(runs, seeds),
        "multi": lambda: multi(runs, seeds),
        "tiny": lambda: tiny(runs),
    }
    for name, job in jobs.items():
        if args.only and name not in args.only:
            continue
        print(f"evaluating {name} ...", flush=True)
        (results / f"{name}.json").write_text(json.dumps(job(), default=float))


if __name__ == "__main__":
    main()
