"""Stochastic order arrivals: a non-homogeneous Poisson process sampled by thinning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import TIERS


@dataclass(frozen=True)
class Order:
    order_id: int
    arrival_time: float
    items: tuple[int, ...]
    item_weights: tuple[int, ...]
    tier: int
    value: float
    dock: int
    vip: bool = False

    @property
    def weight(self) -> int:
        return sum(self.item_weights)

    @property
    def tier_name(self) -> str:
        return TIERS[self.tier]


class OrderGenerator:
    """Sample orders with rate ``λ(t) = λ_base·(1 + a·sin(2πt/T − π/2)) + spike(t)``.

    The sinusoid gives a quiet start, a midday peak and a quiet end of shift; the
    Gaussian ``spike`` adds one rush at a random time. Arrivals use Lewis–Shedler
    thinning against the constant bound ``λ_max``.
    """

    def __init__(
        self,
        shelf_count: int,
        dock_count: int = 3,
        base_rate: float = 0.025,
        horizon: int = 720,
        seed: int | np.random.Generator | None = None,
        *,
        rate_amplitude: float = 0.8,
        spike_height: float = 1.5,
        spike_width: float = 0.03,
        items_range: tuple[int, int] = (1, 3),
        item_weight_range: tuple[int, int] = (1, 3),
        tier_probs: tuple[float, float, float] = (0.2, 0.6, 0.2),
        value_mu: tuple[float, float, float] = (2.7, 2.1, 1.6),
        value_sigma: float = 0.4,
        zipf_exponent: float = 1.0,
        vip_prob: float = 0.0,
        vip_value_mult: float = 5.0,
        max_orders: int | None = None,
    ) -> None:
        if shelf_count < 1 or dock_count < 1 or base_rate < 0 or not 0 <= rate_amplitude <= 1:
            raise ValueError("shelf_count and dock_count must be positive; base_rate >= 0; amplitude in [0, 1]")
        self.shelf_count = shelf_count
        self.dock_count = dock_count
        self.base_rate = base_rate
        self.horizon = horizon
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.rate_amplitude = rate_amplitude
        self.spike_height = spike_height
        self.spike_width = spike_width * horizon
        self.spike_center = float(self.rng.uniform(0.15, 0.85) * horizon)
        self.items_range = items_range
        self.item_weight_range = item_weight_range
        self.tier_probs = np.asarray(tier_probs, dtype=float) / sum(tier_probs)
        self.value_mu = value_mu
        self.value_sigma = value_sigma
        self.vip_prob = vip_prob
        self.vip_value_mult = vip_value_mult
        self.max_orders = max_orders
        # Hot shelves: Zipf popularity over a random permutation of shelf ranks.
        ranks = self.rng.permutation(shelf_count) + 1
        popularity = 1.0 / ranks.astype(float) ** zipf_exponent
        self.shelf_probs = popularity / popularity.sum()

    def rate(self, time: float) -> float:
        phase = 2 * math.pi * time / self.horizon - math.pi / 2
        base = self.base_rate * (1.0 + self.rate_amplitude * math.sin(phase))
        spike = self.base_rate * self.spike_height * math.exp(-0.5 * ((time - self.spike_center) / self.spike_width) ** 2)
        return base + spike

    @property
    def max_rate(self) -> float:
        return self.base_rate * (1.0 + self.rate_amplitude + self.spike_height)

    def generate(self) -> list[Order]:
        orders: list[Order] = []
        bound = self.max_rate
        if bound <= 0:
            return orders
        time = 0.0
        while True:
            time += float(self.rng.exponential(1.0 / bound))
            if time >= self.horizon or (self.max_orders is not None and len(orders) >= self.max_orders):
                return orders
            if self.rng.random() * bound <= self.rate(time):
                orders.append(self._sample_order(len(orders), time))

    def _sample_order(self, order_id: int, time: float) -> Order:
        rng = self.rng
        count = min(int(rng.integers(self.items_range[0], self.items_range[1] + 1)), self.shelf_count)
        items = tuple(int(i) for i in rng.choice(self.shelf_count, size=count, replace=False, p=self.shelf_probs))
        weights = tuple(int(w) for w in rng.integers(self.item_weight_range[0], self.item_weight_range[1] + 1, size=count))
        vip = bool(rng.random() < self.vip_prob)
        tier = 0 if vip else int(rng.choice(3, p=self.tier_probs))
        value = float(rng.lognormal(self.value_mu[tier], self.value_sigma))
        if vip:
            value *= self.vip_value_mult
        dock = int(rng.integers(self.dock_count))
        return Order(order_id, time, items, weights, tier, value, dock, vip)
