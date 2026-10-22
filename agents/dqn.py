"""RL baseline: Double + dueling DQN with prioritized replay and SMDP n-step returns.

Uses the same attention encoder as the main agent so the comparison isolates the
learning algorithm. Invalid actions are masked both when acting (ε-greedy samples
only valid actions) and in the Double-DQN target ``argmax``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import TorchPolicy, info_value, make_vector_env, to_torch
from .networks import AttentionQNetwork


@dataclass
class DQNConfig:
    total_decisions: int = 1_000_000
    n_envs: int = 16
    buffer_size: int = 200_000
    batch_size: int = 256
    lr: float = 2.5e-4
    gamma: float = 0.995
    n_step: int = 3
    learning_starts: int = 10_000
    gradient_steps: int = 1
    target_update_decisions: int = 8_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_eps: float = 1e-3
    reward_scale: float = 0.1
    max_grad_norm: float = 10.0
    double: bool = True
    async_envs: bool = True


class SumTree:
    """Binary sum tree over ``capacity`` leaves for O(log n) proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        self.tree = np.zeros(2 * self.size)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(np.asarray(indices) + self.size, priorities):
            delta = priority - self.tree[index]
            while index >= 1:
                self.tree[index] += delta
                index //= 2

    def find(self, values: np.ndarray) -> np.ndarray:
        """Leaf index for each cumulative ``value`` (vectorised descent)."""
        index = np.ones(len(values), dtype=np.int64)
        values = values.copy()
        while index[0] < self.size:
            left = 2 * index
            go_right = values > self.tree[left]
            values -= self.tree[left] * go_right
            index = left + go_right
        return index - self.size

    def leaf(self, indices: np.ndarray) -> np.ndarray:
        return self.tree[np.asarray(indices) + self.size]


class PrioritizedReplay:
    def __init__(self, capacity: int, example_obs: dict[str, np.ndarray], alpha: float, eps: float) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.obs = {k: np.zeros((capacity, *v.shape), dtype=v.dtype) for k, v in example_obs.items()}
        self.next_obs = {k: np.zeros((capacity, *v.shape), dtype=v.dtype) for k, v in example_obs.items()}
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, ret, discount, next_obs) -> None:
        i = self.pos
        for k in self.obs:
            self.obs[k][i] = obs[k]
            self.next_obs[k][i] = next_obs[k]
        self.actions[i], self.returns[i], self.discounts[i] = action, ret, discount
        self.tree.update([i], [self.max_priority**self.alpha])
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch: int, beta: float, rng: np.random.Generator):
        segment = self.tree.total / batch
        values = (np.arange(batch) + rng.random(batch)) * segment
        indices = np.minimum(self.tree.find(values), len(self) - 1)
        probs = self.tree.leaf(indices) / self.tree.total
        weights = (len(self) * np.maximum(probs, 1e-12)) ** (-beta)
        weights /= weights.max()
        pick = lambda store: {k: v[indices] for k, v in store.items()}  # noqa: E731
        return (
            indices,
            pick(self.obs),
            self.actions[indices],
            self.returns[indices],
            self.discounts[indices],
            pick(self.next_obs),
            weights,
        )

    def update_priorities(self, indices, td_errors) -> None:
        priorities = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.update(indices, priorities**self.alpha)


class NStepBuffer:
    """Per-env n-step accumulator: ``R = Σ_k γ^{τ_0+…+τ_{k-1}} r_k`` and bootstrap discount ``γ^{Στ}``."""

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self.items: deque = deque()

    def push(self, obs, action, reward, tau, next_obs, done) -> list[tuple]:
        self.items.append((obs, action, reward, tau))
        out = []
        if done:
            while self.items:
                out.append(self._emit(next_obs, terminal=True))
        elif len(self.items) == self.n:
            out.append(self._emit(next_obs, terminal=False))
        return out

    def _emit(self, next_obs, terminal: bool) -> tuple:
        ret, elapsed = 0.0, 0.0
        for _, _, reward, tau in self.items:
            ret += self.gamma**elapsed * reward
            elapsed += tau
        obs, action, _, _ = self.items.popleft()
        return obs, action, ret, 0.0 if terminal else self.gamma**elapsed, next_obs


class DQNTrainer:
    def __init__(self, probe_env, env_config, cfg: DQNConfig, net_kwargs: dict, seed: int, logger, curriculum, evaluator) -> None:
        self.cfg = cfg
        self.seed = seed
        self.logger = logger
        self.curriculum = curriculum
        self.evaluator = evaluator
        kwargs = {"n_docks": probe_env.dock_count, "use_robots": "robots" in probe_env.observation_space.spaces, **net_kwargs}
        self.net = AttentionQNetwork(**kwargs)
        self.target = AttentionQNetwork(**kwargs)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.envs = make_vector_env(env_config, cfg.n_envs, cfg.async_envs)
        if curriculum.level is not None:
            self.envs.call("set_level", curriculum.level)
        self.rng = np.random.default_rng(seed)
        self.decisions = 0
        self.ticks = 0.0
        self.episodes: list[dict] = []

    def epsilon(self) -> float:
        progress = min(1.0, self.decisions / (self.cfg.eps_fraction * self.cfg.total_decisions))
        return self.cfg.eps_start + progress * (self.cfg.eps_end - self.cfg.eps_start)

    def train(self) -> None:
        cfg = self.cfg
        n = cfg.n_envs
        obs, _ = self.envs.reset(seed=[self.seed * 1000 + i for i in range(n)])
        single = {k: v[0] for k, v in obs.items()}
        replay = PrioritizedReplay(cfg.buffer_size, single, cfg.per_alpha, cfg.per_eps)
        nsteps = [NStepBuffer(cfg.n_step, cfg.gamma) for _ in range(n)]
        total_updates = max(1, cfg.total_decisions // 4096)
        last_update_index, last_target = 0, 0
        metrics: dict[str, float] = {}
        start = time.time()
        while self.decisions < cfg.total_decisions:
            masks = obs["action_mask"].astype(bool)
            with torch.no_grad():
                q, _ = self.net(to_torch(obs), mask=torch.as_tensor(masks))
                greedy = q.masked_fill(~torch.as_tensor(masks), -1e9).argmax(-1).numpy()
            explore = self.rng.random(n) < self.epsilon()
            actions = np.array([self.rng.choice(np.flatnonzero(m)) if e else g for m, e, g in zip(masks, explore, greedy)])
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
            done = terminated | truncated
            for i in range(n):
                tau = float(info_value(infos, "tau", i, done[i]))
                self.ticks += tau
                # On termination the bootstrap discount is 0, so the (already reset) next obs is never used.
                nxt = {k: v[i] for k, v in next_obs.items()}
                for item in nsteps[i].push(
                    {k: v[i] for k, v in obs.items()}, actions[i], rewards[i] * cfg.reward_scale, tau, nxt, done[i]
                ):
                    replay.add(*item)
                if done[i]:
                    stats = info_value(infos, "episode_stats", i, True)
                    self.episodes.append({k: float(v) for k, v in stats.items()})
                    self.curriculum.record(float(stats["on_time_pct"]))
            obs = next_obs
            self.decisions += n

            if len(replay) >= cfg.learning_starts:
                for _ in range(cfg.gradient_steps):
                    metrics = self._learn(replay)
                if self.decisions - last_target >= cfg.target_update_decisions:
                    self.target.load_state_dict(self.net.state_dict())
                    last_target = self.decisions

            update_index = self.decisions // 4096
            if update_index > last_update_index:
                last_update_index = update_index
                log = dict(metrics)
                log["charts/epsilon"] = self.epsilon()
                log["charts/decisions_per_s"] = self.decisions / (time.time() - start)
                if self.episodes:
                    recent = self.episodes[-100:]
                    for key in (
                        "return",
                        "on_time_pct",
                        "express_on_time_pct",
                        "revenue_captured_pct",
                        "trucks_missed",
                        "battery_failures",
                        "items_per_trip",
                    ):
                        log[f"episode/{key}"] = float(np.mean([e[key] for e in recent]))
                if self.curriculum.step():
                    self.envs.call("set_level", self.curriculum.level)
                log["curriculum/level"] = -1 if self.curriculum.level is None else self.curriculum.level
                self.logger.scalars(log, self.decisions, self.ticks)
                if self.evaluator.due(update_index, total_updates):
                    self.evaluator.run(self, update_index)
        self.evaluator.run(self, total_updates)
        self.envs.close()

    def _learn(self, replay: PrioritizedReplay) -> dict[str, float]:
        cfg = self.cfg
        beta = cfg.per_beta_start + (1.0 - cfg.per_beta_start) * min(1.0, self.decisions / cfg.total_decisions)
        indices, obs, actions, returns, discounts, next_obs, weights = replay.sample(cfg.batch_size, beta, self.rng)
        obs_t, next_t = to_torch(obs), to_torch(next_obs)
        next_mask = next_t["action_mask"]
        with torch.no_grad():
            target_q, _ = self.target(next_t, mask=next_mask)
            if cfg.double:
                online_next, _ = self.net(next_t, mask=next_mask)
                best = online_next.masked_fill(~next_mask, -1e9).argmax(-1, keepdim=True)
                next_value = target_q.gather(1, best).squeeze(1)
            else:
                next_value = target_q.masked_fill(~next_mask, -1e9).max(-1).values
            target = torch.as_tensor(returns) + torch.as_tensor(discounts) * next_value
        self.net.train()
        q, _ = self.net(obs_t, mask=obs_t["action_mask"])
        q_taken = q.gather(1, torch.as_tensor(actions).unsqueeze(1)).squeeze(1)
        td = q_taken - target
        loss = (torch.as_tensor(weights, dtype=torch.float32) * F.smooth_l1_loss(q_taken, target, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        replay.update_priorities(indices, td.detach().numpy())
        return {"losses/td_loss": loss.item(), "losses/q_mean": q_taken.mean().item(), "charts/per_beta": beta}

    def policy(self) -> TorchPolicy:
        return TorchPolicy(self.net, "q", deterministic=True)
