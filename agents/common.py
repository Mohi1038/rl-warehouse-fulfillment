"""Shared agent utilities: tensors, vector envs, normalisation, checkpoints, batched evaluation."""

from __future__ import annotations

import csv
import functools
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from warehouse_env import EnvConfig, WarehouseEnv

VAL_SEEDS = tuple(range(50_000, 50_020))
EVAL_SEEDS = tuple(range(100_000, 100_100))


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def to_torch(obs: dict[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    out = {}
    for key, value in obs.items():
        dtype = torch.bool if key == "action_mask" else torch.float32
        out[key] = torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    return out


def stack_obs(observations: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}


def make_env(config: EnvConfig, render_mode: str | None = None) -> WarehouseEnv:
    return WarehouseEnv(config, render_mode=render_mode)


def make_vector_env(config: EnvConfig, n_envs: int, asynchronous: bool = True) -> gym.vector.VectorEnv:
    fns = [functools.partial(make_env, config) for _ in range(n_envs)]
    mode = gym.vector.AutoresetMode.SAME_STEP
    if asynchronous and n_envs > 1:
        return gym.vector.AsyncVectorEnv(fns, autoreset_mode=mode, context="fork")
    return gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)


def info_value(infos: dict, key: str, index: int, done: bool):
    """Read a per-env info entry, preferring the pre-reset ``final_info`` for finished episodes."""
    source = infos["final_info"] if done and "final_info" in infos else infos
    value = source[key]
    if isinstance(value, dict):
        return {k: v[index] for k, v in value.items() if not k.startswith("_")}
    return value[index]


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = np.mean(x, axis=0), np.var(x, axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}


class TorchPolicy:
    """Wraps an actor-critic (``logits, value, attention``) or Q-network (``q, attention``) for acting."""

    def __init__(self, net: torch.nn.Module, kind: str, deterministic: bool = True, use_mask: bool = True, seed: int = 0) -> None:
        self.net = net.eval()
        self.kind = kind
        self.deterministic = deterministic
        self.use_mask = use_mask
        self.generator = torch.Generator().manual_seed(seed)
        self.last_attention: np.ndarray | None = None

    @torch.no_grad()
    def act_batch(self, obs: dict[str, np.ndarray], need_weights: bool = False) -> np.ndarray:
        t = to_torch(obs)
        mask = t["action_mask"] if self.use_mask else None
        if self.kind == "q":
            scores, attention = self.net(t, mask=t["action_mask"], need_weights=need_weights)
        else:
            scores, _, attention = self.net(t, need_weights=need_weights)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        self.last_attention = None if attention is None else attention.numpy()
        if self.deterministic or self.kind == "q":
            return scores.argmax(-1).numpy()
        probs = torch.softmax(scores, -1)
        return torch.multinomial(probs, 1, generator=self.generator).squeeze(-1).numpy()

    def act(self, obs: dict[str, np.ndarray]) -> int:
        return int(self.act_batch({k: v[None] for k, v in obs.items()})[0])

    def attention(self, obs: dict[str, np.ndarray]) -> np.ndarray | None:
        self.act_batch({k: v[None] for k, v in obs.items()}, need_weights=True)
        return None if self.last_attention is None else self.last_attention[0]


def save_checkpoint(
    path: str | Path, net: torch.nn.Module, algo: str, net_kind: str, config: dict, extra: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": algo,
            "net_kind": net_kind,
            "net_kwargs": net.kwargs,
            "state_dict": net.state_dict(),
            "config": config,
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def policy_from_checkpoint(path: str | Path, deterministic: bool = True) -> tuple[TorchPolicy, dict]:
    from .networks import build_network

    ckpt = load_checkpoint(path)
    net = build_network(ckpt["net_kind"], **ckpt["net_kwargs"])
    net.load_state_dict(ckpt["state_dict"])
    kind = "q" if ckpt["net_kind"] == "attention_q" else "actor_critic"
    use_mask = ckpt["config"].get("algo_config", {}).get("use_mask", True)
    return TorchPolicy(net, kind, deterministic=deterministic, use_mask=use_mask), ckpt


def evaluate_policy(
    act_batch: Callable[[dict[str, np.ndarray]], np.ndarray],
    config: EnvConfig,
    seeds: Sequence[int] = VAL_SEEDS,
    level: int | None = None,
    on_step: Callable | None = None,
) -> list[dict]:
    """Run one episode per seed in lock-step, batching policy calls across live episodes."""
    envs = [WarehouseEnv(config) for _ in seeds]
    observations = []
    for env, seed in zip(envs, seeds):
        env.set_level(level)
        observations.append(env.reset(seed=int(seed))[0])
    live = list(range(len(envs)))
    results: list[dict | None] = [None] * len(envs)
    while live:
        actions = act_batch(stack_obs([observations[i] for i in live]))
        still = []
        for i, action in zip(live, actions):
            obs, reward, terminated, truncated, info = envs[i].step(int(action))
            if on_step is not None:
                on_step(i, envs[i], observations[i], int(action), reward, info)
            observations[i] = obs
            if terminated or truncated:
                stats = info["episode_stats"]
                stats["seed"] = int(seeds[i])
                stats.update({f"actions_{k}": v for k, v in envs[i].stats["action_counts"].items()})
                results[i] = stats
            else:
                still.append(i)
        live = still
    return results


def baseline_act_batch(policy) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    def act_batch(obs):
        n = len(obs["global"])
        return np.array([policy.act({k: v[i] for k, v in obs.items()}) for i in range(n)])

    return act_batch


def summarize(results: list[dict]) -> dict[str, float]:
    keys = [k for k in results[0] if k != "seed"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None

    def log(self, row: dict) -> None:
        new = self._fields is None
        if new:
            self._fields = list(row)
        with self.path.open("a" if not new else "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)
