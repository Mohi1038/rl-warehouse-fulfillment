"""PPO for the warehouse SMDP, written from scratch (CleanRL-style).

Differences from vanilla PPO:
* invalid actions are masked out of the categorical distribution;
* advantages use SMDP-correct GAE, discounting each transition by ``γ^τ``:
  ``δ_t = r_t + γ^{τ_t} V(s_{t+1})(1−d_t) − V(s_t)``, ``Â_t = δ_t + (γλ)^{τ_t}(1−d_t)Â_{t+1}``;
* Dict observations with a variable-size order set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from curriculum import Curriculum

from .common import RunningMeanStd, TorchPolicy, evaluate_policy, info_value, make_vector_env, summarize, to_torch


@dataclass
class PPOConfig:
    total_decisions: int = 2_000_000
    n_envs: int = 16
    n_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 1024
    lr: float = 3e-4
    anneal_lr: bool = True
    clip_coef: float = 0.2
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    lambda_per_tick: bool = True
    smdp_discount: bool = True
    target_kl: float | None = 0.02
    norm_adv: bool = True
    clip_vloss: bool = True
    reward_norm: bool = True
    use_mask: bool = True
    async_envs: bool = True


def smdp_gae(rewards, values, dones, taus, last_value, gamma, lam, smdp=True, lambda_per_tick=True):
    """Generalised advantage estimation for temporally-extended actions. Arrays are ``[T, N]``."""
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last = np.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        discount = gamma ** taus[t] if smdp else np.full_like(taus[t], gamma)
        trace = lam ** taus[t] if (smdp and lambda_per_tick) else np.full_like(taus[t], lam)
        delta = rewards[t] + discount * next_value * nonterminal - values[t]
        last = delta + discount * trace * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, net: nn.Module, env_config, cfg: PPOConfig, seed: int, logger, curriculum: Curriculum, evaluator) -> None:
        self.net = net
        self.cfg = cfg
        self.seed = seed
        self.logger = logger
        self.curriculum = curriculum
        self.evaluator = evaluator
        self.envs = make_vector_env(env_config, cfg.n_envs, cfg.async_envs)
        if curriculum.level is not None:
            self.envs.call("set_level", curriculum.level)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.return_rms = RunningMeanStd()
        self.discounted = np.zeros(cfg.n_envs)
        self.decisions = 0
        self.ticks = 0.0
        self.episodes: list[dict] = []

    def _dist(self, logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        if self.cfg.use_mask:
            logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    def train(self) -> None:
        cfg = self.cfg
        n, s = cfg.n_envs, cfg.n_steps
        batch = n * s
        updates = max(1, cfg.total_decisions // batch)
        obs, _ = self.envs.reset(seed=[self.seed * 1000 + i for i in range(n)])
        keys = list(obs)
        buffers = {k: np.zeros((s, *obs[k].shape), dtype=obs[k].dtype) for k in keys}
        actions = np.zeros((s, n), dtype=np.int64)
        logprobs = np.zeros((s, n), dtype=np.float32)
        values = np.zeros((s, n), dtype=np.float32)
        rewards = np.zeros((s, n), dtype=np.float32)
        dones = np.zeros((s, n), dtype=np.float32)
        taus = np.zeros((s, n), dtype=np.float32)
        start = time.time()

        for update in range(1, updates + 1):
            frac = 1.0 - (update - 1) / updates
            if cfg.anneal_lr:
                self.optimizer.param_groups[0]["lr"] = frac * cfg.lr
            ent_coef = cfg.ent_coef_end + (cfg.ent_coef_start - cfg.ent_coef_end) * frac
            term_sums: dict[str, float] = {}

            self.net.eval()
            for step in range(s):
                for k in keys:
                    buffers[k][step] = obs[k]
                with torch.no_grad():
                    t = to_torch(obs)
                    logits, value, _ = self.net(t)
                    dist = self._dist(logits, t["action_mask"])
                    action = dist.sample()
                actions[step] = action.numpy()
                logprobs[step] = dist.log_prob(action).numpy()
                values[step] = value.numpy()
                obs, reward, terminated, truncated, infos = self.envs.step(actions[step])
                done = terminated | truncated
                tau = np.array([info_value(infos, "tau", i, done[i]) for i in range(n)], dtype=np.float32)
                for i in range(n):
                    for name, v in info_value(infos, "reward_terms", i, done[i]).items():
                        term_sums[name] = term_sums.get(name, 0.0) + float(v)
                    if done[i]:
                        stats = info_value(infos, "episode_stats", i, True)
                        self.episodes.append({k: float(v) for k, v in stats.items()})
                        self.curriculum.record(float(stats["on_time_pct"]))
                taus[step] = tau
                dones[step] = done
                if cfg.reward_norm:
                    discount = cfg.gamma**tau if cfg.smdp_discount else cfg.gamma
                    self.discounted = self.discounted * discount + reward
                    self.return_rms.update(self.discounted)
                    self.discounted[done] = 0.0
                    rewards[step] = reward / np.sqrt(self.return_rms.var + 1e-8)
                else:
                    rewards[step] = reward
                self.decisions += n
                self.ticks += float(tau.sum())

            with torch.no_grad():
                _, last_value, _ = self.net(to_torch(obs))
            advantages, returns = smdp_gae(
                rewards,
                values,
                dones,
                taus,
                last_value.numpy(),
                cfg.gamma,
                cfg.gae_lambda,
                cfg.smdp_discount,
                cfg.lambda_per_tick,
            )
            metrics = self._update(buffers, actions, logprobs, values, advantages, returns, ent_coef, batch)
            metrics.update({f"reward_terms/{k}": v / n for k, v in term_sums.items()})
            metrics["charts/decisions_per_s"] = self.decisions / (time.time() - start)
            metrics["charts/ent_coef"] = ent_coef
            metrics["charts/lr"] = self.optimizer.param_groups[0]["lr"]
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
                    metrics[f"episode/{key}"] = float(np.mean([e[key] for e in recent]))
            if self.curriculum.step():
                self.envs.call("set_level", self.curriculum.level)
            metrics["curriculum/level"] = -1 if self.curriculum.level is None else self.curriculum.level
            metrics["curriculum/on_time"] = self.curriculum.on_time
            self.logger.scalars(metrics, self.decisions, self.ticks)
            if self.evaluator.due(update, updates):
                self.evaluator.run(self, update)
        self.envs.close()

    def _update(self, buffers, actions, logprobs, values, advantages, returns, ent_coef, batch):
        cfg = self.cfg
        self.net.train()
        flat = {k: torch.as_tensor(v.reshape(batch, *v.shape[2:])) for k, v in buffers.items()}
        flat["action_mask"] = flat["action_mask"].bool()
        b_actions = torch.as_tensor(actions.reshape(-1))
        b_logprobs = torch.as_tensor(logprobs.reshape(-1))
        b_values = torch.as_tensor(values.reshape(-1))
        b_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32)
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
        clipfracs, approx_kl = [], torch.tensor(0.0)
        indices = np.arange(batch)
        stop = False
        for _ in range(cfg.epochs):
            np.random.shuffle(indices)
            for begin in range(0, batch, cfg.minibatch_size):
                mb = torch.as_tensor(indices[begin : begin + cfg.minibatch_size])
                mb_obs = {k: v[mb] for k, v in flat.items()}
                logits, new_value, _ = self.net(mb_obs)
                dist = self._dist(logits, mb_obs["action_mask"])
                new_logprob = dist.log_prob(b_actions[mb])
                log_ratio = new_logprob - b_logprobs[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                adv = b_adv[mb]
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                if cfg.clip_vloss:
                    clipped = b_values[mb] + (new_value - b_values[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max((new_value - b_returns[mb]) ** 2, (clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss - ent_coef * entropy + cfg.vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop = True
                break
        y_pred, y_true = b_values.numpy(), b_returns.numpy()
        var_y = np.var(y_true)
        return {
            "losses/policy": pg_loss.item(),
            "losses/value": v_loss.item(),
            "losses/entropy": entropy.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "losses/explained_variance": float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y),
            "losses/early_stop": float(stop),
        }

    def policy(self) -> TorchPolicy:
        return TorchPolicy(self.net, "actor_critic", deterministic=True, use_mask=self.cfg.use_mask)


def evaluate_trainer_policy(policy: TorchPolicy, env_config, seeds) -> dict:
    return summarize(evaluate_policy(policy.act_batch, env_config, seeds))
