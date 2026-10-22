"""Policy/value networks over the Dict observation.

* ``AttentionActorCritic`` — permutation-equivariant order encoder ([CLS] + order tokens
  through a Transformer), a pointer head scoring each order slot and a small head for
  DELIVER/CHARGE/WAIT. Works for any number of order slots without retraining.
* ``MLPActorCritic`` — flatten + MLP ablation; slots are sorted by slack first because an
  MLP is not permutation invariant, and the PICK logits are scattered back afterwards.
* ``AttentionQNetwork`` — dueling Q-network on the same encoder for DQN.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from warehouse_env.env import GLOBAL_DIM, ORDER_DIM, ROBOT_DIM

NEG_INF = -1e9
SLACK_FEATURE = 8


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(sizes: list[int], out_gain: float = np.sqrt(2), activation=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        layers.append(layer_init(nn.Linear(sizes[i], sizes[i + 1]), out_gain if last else np.sqrt(2)))
        if not last:
            layers.append(activation())
    return nn.Sequential(*layers)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return logits if mask is None else logits.masked_fill(~mask, NEG_INF)


class EncoderLayer(nn.Module):
    """Pre-norm Transformer layer that can also return attention weights (for the overlay)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.ReLU(), nn.Linear(ff_mult * d_model, d_model))

    def forward(self, x: torch.Tensor, pad: torch.Tensor, need_weights: bool = False):
        h = self.norm1(x)
        out, weights = self.attn(h, h, h, key_padding_mask=pad, need_weights=need_weights, average_attn_weights=True)
        x = x + out
        return x + self.ff(self.norm2(x)), weights


class AttentionEncoder(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, use_robots: bool = False) -> None:
        super().__init__()
        self.use_robots = use_robots
        self.global_net = mlp([GLOBAL_DIM, d_model, d_model])
        self.order_net = mlp([ORDER_DIM, d_model, d_model])
        if use_robots:
            self.robot_net = mlp([ROBOT_DIM, d_model, d_model])
            self.type_embed = nn.Parameter(torch.zeros(3, d_model))
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: dict[str, torch.Tensor], need_weights: bool = False):
        orders = obs["orders"]
        batch, n_orders, _ = orders.shape
        valid = orders[..., 0] > 0.5
        tokens = [self.global_net(obs["global"]).unsqueeze(1), self.order_net(orders)]
        pads = [torch.zeros(batch, 1, dtype=torch.bool, device=orders.device), ~valid]
        if self.use_robots:
            robots = obs["robots"]
            tokens[0] = tokens[0] + self.type_embed[0]
            tokens[1] = tokens[1] + self.type_embed[1]
            tokens.append(self.robot_net(robots) + self.type_embed[2])
            pads.append(robots[..., 0] < 0.5)
        x = torch.cat(tokens, dim=1)
        pad = torch.cat(pads, dim=1)
        weights = None
        for i, layer in enumerate(self.layers):
            x, weights = layer(x, pad, need_weights=need_weights and i == len(self.layers) - 1)
        x = self.norm(x)
        h_cls, h_orders = x[:, 0], x[:, 1 : 1 + n_orders]
        cls_attention = weights[:, 0, 1 : 1 + n_orders] if weights is not None else None
        return h_cls, h_orders, valid, cls_attention


class AttentionActorCritic(nn.Module):
    def __init__(self, n_docks: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, use_robots: bool = False) -> None:
        super().__init__()
        self.kwargs = {"n_docks": n_docks, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "use_robots": use_robots}
        self.encoder = AttentionEncoder(d_model, n_layers, n_heads, use_robots)
        self.pointer = mlp([2 * d_model, d_model, 1], out_gain=0.01)
        self.other = mlp([d_model, d_model, n_docks + 2], out_gain=0.01)
        self.value_head = mlp([2 * d_model, d_model, 1], out_gain=1.0)

    def forward(self, obs, need_weights: bool = False):
        h_cls, h_orders, valid, attention = self.encoder(obs, need_weights)
        cls = h_cls.unsqueeze(1).expand_as(h_orders)
        pick = self.pointer(torch.cat([h_orders, cls], dim=-1)).squeeze(-1)
        logits = torch.cat([pick, self.other(h_cls)], dim=-1)
        weights = valid.float().unsqueeze(-1)
        pooled = (h_orders * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        value = self.value_head(torch.cat([h_cls, pooled], dim=-1)).squeeze(-1)
        return logits, value, attention


class MLPActorCritic(nn.Module):
    def __init__(self, n_docks: int, n_orders: int, hidden: int = 256, use_robots: bool = False, sort_by_slack: bool = True) -> None:
        super().__init__()
        self.kwargs = {"n_docks": n_docks, "n_orders": n_orders, "hidden": hidden, "use_robots": use_robots, "sort_by_slack": sort_by_slack}
        self.n_orders = n_orders
        self.use_robots = use_robots
        self.sort_by_slack = sort_by_slack
        in_dim = GLOBAL_DIM + n_orders * ORDER_DIM + (3 * ROBOT_DIM if use_robots else 0)
        self.trunk = mlp([in_dim, hidden, hidden])
        self.trunk.append(nn.ReLU())
        self.policy_head = layer_init(nn.Linear(hidden, n_orders + n_docks + 2), 0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), 1.0)

    def forward(self, obs, need_weights: bool = False):
        orders = obs["orders"]
        if orders.shape[1] != self.n_orders:
            raise ValueError(f"MLP policy was built for {self.n_orders} order slots, got {orders.shape[1]}")
        perm = None
        if self.sort_by_slack:
            # slack first; age and value break ties so equal-slack orders sort the same way under any slot permutation
            slack = orders[..., SLACK_FEATURE] - 1e-3 * orders[..., 13] - 1e-5 * orders[..., 1]
            key = torch.where(orders[..., 0] > 0.5, slack, torch.full_like(slack, 1e6))
            perm = key.argsort(dim=1, stable=True)
            orders = orders.gather(1, perm.unsqueeze(-1).expand_as(orders))
        parts = [obs["global"], orders.flatten(1)]
        if self.use_robots:
            parts.append(obs["robots"].flatten(1))
        h = self.trunk(torch.cat(parts, dim=-1))
        logits = self.policy_head(h)
        if perm is not None:
            pick = torch.empty_like(logits[:, : self.n_orders]).scatter_(1, perm, logits[:, : self.n_orders])
            logits = torch.cat([pick, logits[:, self.n_orders :]], dim=-1)
        return logits, self.value_head(h).squeeze(-1), None


class AttentionQNetwork(nn.Module):
    """Dueling Q: ``Q = V + A − mean_valid(A)`` with pointer-style advantages per order slot."""

    def __init__(self, n_docks: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, use_robots: bool = False) -> None:
        super().__init__()
        self.kwargs = {"n_docks": n_docks, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "use_robots": use_robots}
        self.encoder = AttentionEncoder(d_model, n_layers, n_heads, use_robots)
        self.pointer = mlp([2 * d_model, d_model, 1], out_gain=1.0)
        self.other = mlp([d_model, d_model, n_docks + 2], out_gain=1.0)
        self.value_head = mlp([2 * d_model, d_model, 1], out_gain=1.0)

    def forward(self, obs, mask: torch.Tensor | None = None, need_weights: bool = False):
        h_cls, h_orders, valid, attention = self.encoder(obs, need_weights)
        cls = h_cls.unsqueeze(1).expand_as(h_orders)
        advantage = torch.cat([self.pointer(torch.cat([h_orders, cls], dim=-1)).squeeze(-1), self.other(h_cls)], dim=-1)
        weights = valid.float().unsqueeze(-1)
        pooled = (h_orders * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        value = self.value_head(torch.cat([h_cls, pooled], dim=-1))
        if mask is None:
            mean_adv = advantage.mean(-1, keepdim=True)
        else:
            m = mask.float()
            mean_adv = (advantage * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1.0)
        return value + advantage - mean_adv, attention


NETWORKS = {"attention": AttentionActorCritic, "mlp": MLPActorCritic, "attention_q": AttentionQNetwork}


def build_network(kind: str, **kwargs) -> nn.Module:
    return NETWORKS[kind](**kwargs)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
