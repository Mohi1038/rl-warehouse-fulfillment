"""Train an agent from a YAML config.

    python train.py --config configs/ppo_attn.yaml --seed 0
    python train.py --config configs/ppo_attn.yaml --seed 1 --set algo_config.total_decisions=200000

A run writes ``runs/<name>/s<seed>/`` (TensorBoard events, ``eval.csv``, ``best.pt``,
``last.pt``, ``config.yaml``). ``best.pt`` is selected on validation seeds that are
disjoint from the held-out evaluation seeds used by ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.common import VAL_SEEDS, CSVLogger, evaluate_policy, save_checkpoint, seed_everything, summarize
from curriculum import Curriculum, CurriculumConfig
from warehouse_env import EnvConfig, WarehouseEnv, tiny_config

ALGOS = ("ppo_attn", "ppo_mlp", "dqn", "tiny_q")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    base = config.pop("extends", None)
    if base:
        parent = load_config(Path(path).parent / base)
        config = deep_merge(parent, config)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        target = config
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = yaml.safe_load(raw)
    return config


def deep_merge(base: dict, update: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def env_config_from(config: dict, key: str = "env") -> EnvConfig:
    section = dict(config.get(key) or {})
    preset = section.pop("preset", None)
    if preset == "tiny":
        return tiny_config(**{k: tuple(v) if isinstance(v, list) else v for k, v in section.items()})
    return EnvConfig.from_dict(section)


def eval_env_config(config: dict) -> EnvConfig:
    """Nominal evaluation setting: the training env without randomisation, plus ``eval_env`` overrides."""
    overrides = config.get("eval_env") or {}
    merged = deep_merge(config.get("env") or {}, overrides)
    merged.setdefault("domain_randomization", False)
    if "domain_randomization" not in overrides:
        merged["domain_randomization"] = False
    return env_config_from({"env": merged})


def dataclass_from(cls, data: dict | None):
    data = dict(data or {})
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


class Logger:
    def __init__(self, run_dir: Path) -> None:
        self.writer = SummaryWriter(str(run_dir))
        self.last_print = 0.0

    def scalars(self, metrics: dict, decisions: int, ticks: float) -> None:
        for key, value in metrics.items():
            if value is not None and np.isfinite(value):
                self.writer.add_scalar(key, value, decisions)
        self.writer.add_scalar("charts/ticks", ticks, decisions)
        if time.time() - self.last_print > 20:
            self.last_print = time.time()
            shown = {k.split("/")[-1]: round(v, 3) for k, v in metrics.items() if k.startswith(("episode/", "curriculum/level"))}
            print(f"[{decisions:>9d} decisions] {shown}", flush=True)


class Evaluator:
    """Periodic deterministic evaluation on validation seeds; keeps the best checkpoint."""

    def __init__(self, run_dir: Path, config: dict, env_config: EnvConfig, net_kind: str, every: int, episodes: int) -> None:
        self.run_dir = run_dir
        self.config = config
        self.env_config = env_config
        self.net_kind = net_kind
        self.every = every
        self.seeds = VAL_SEEDS[:episodes]
        self.best = -np.inf
        self.metric = "task_return"  # return without PBRS, comparable across shaping ablations
        self.csv = CSVLogger(run_dir / "eval.csv")

    def due(self, index: int, total: int) -> bool:
        return index % self.every == 0 or index == total

    def run(self, trainer, index: int) -> dict:
        summary = summarize(evaluate_policy(trainer.policy().act_batch, self.env_config, self.seeds))
        trainer.net.train()
        episode = {("episode_decisions" if k == "decisions" else k): v for k, v in summary.items()}
        row = {"update": index, "decisions": trainer.decisions, "ticks": trainer.ticks, **episode}
        self.csv.log(row)
        for key, value in summary.items():
            trainer.logger.writer.add_scalar(f"eval/{key}", value, trainer.decisions)
        extra = {"decisions": trainer.decisions, "eval": summary}
        save_checkpoint(self.run_dir / "last.pt", trainer.net, self.config["algo"], self.net_kind, self.config, extra)
        if summary[self.metric] > self.best:
            self.best = summary[self.metric]
            save_checkpoint(self.run_dir / "best.pt", trainer.net, self.config["algo"], self.net_kind, self.config, extra)
        print(
            f"  eval @ {trainer.decisions}: task_return={summary['task_return']:.1f} on_time={summary['on_time_pct']:.1f}% best={self.best:.1f}",
            flush=True,
        )
        return summary


def run(config: dict, seed: int, out_root: str | Path = "runs") -> Path:
    algo = config["algo"]
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")
    name = config.get("name", algo)
    run_dir = Path(out_root) / name / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**config, "seed": seed}, sort_keys=False))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    env_config = env_config_from(config)
    eval_config = eval_env_config(config)
    probe = WarehouseEnv(env_config)

    if algo == "tiny_q":
        from agents.q_learning import QLearningConfig, train_q_learning

        result = train_q_learning(env_config, dataclass_from(QLearningConfig, config.get("algo_config")), seed, run_dir)
        (run_dir / "DONE").write_text(json.dumps(result["q_learning"]))
        return run_dir

    curriculum = Curriculum(dataclass_from(CurriculumConfig, config.get("curriculum", {"enabled": False})))
    logger = Logger(run_dir)
    if algo in ("ppo_attn", "ppo_mlp"):
        from agents import ppo_attn, ppo_mlp
        from agents.ppo import PPOConfig, PPOTrainer

        module = ppo_attn if algo == "ppo_attn" else ppo_mlp
        net = module.build_network(probe, **config.get("network", {}))
        algo_cfg = dataclass_from(PPOConfig, config.get("algo_config"))
        evaluator = Evaluator(
            run_dir, config, eval_config, module.NET_KIND, config.get("eval_every", 50), config.get("eval_episodes", 20)
        )
        trainer = PPOTrainer(net, env_config, algo_cfg, seed, logger, curriculum, evaluator)
    else:
        from agents.dqn import DQNConfig, DQNTrainer

        algo_cfg = dataclass_from(DQNConfig, config.get("algo_config"))
        evaluator = Evaluator(
            run_dir, config, eval_config, "attention_q", config.get("eval_every", 50), config.get("eval_episodes", 20)
        )
        trainer = DQNTrainer(probe, env_config, algo_cfg, config.get("network", {}), seed, logger, curriculum, evaluator)
    print(
        f"{name} seed={seed}: {sum(p.numel() for p in trainer.net.parameters()):,} parameters, config {asdict(algo_cfg)}",
        flush=True,
    )
    trainer.train()
    logger.writer.close()
    (run_dir / "DONE").write_text(json.dumps({"decisions": trainer.decisions, "best": evaluator.best}))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. algo_config.lr=1e-4")
    args = parser.parse_args()
    run_dir = run(load_config(args.config, args.set), args.seed, args.out)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
