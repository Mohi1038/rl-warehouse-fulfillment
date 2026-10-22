import numpy as np
from scipy import integrate

from warehouse_env.orders import OrderGenerator
from warehouse_env.trucks import TruckScheduler


def test_generators_are_seed_deterministic():
    assert OrderGenerator(60, seed=7).generate() == OrderGenerator(60, seed=7).generate()
    assert TruckScheduler(seed=7).generate() == TruckScheduler(seed=7).generate()
    assert OrderGenerator(60, seed=7).generate() != OrderGenerator(60, seed=8).generate()


def test_arrival_histogram_matches_rate():
    """Thinning must reproduce λ(t): binned counts over many shifts match ∫λ within 4σ."""
    horizon, episodes, bins = 720, 3000, 12
    counts = np.zeros(bins)
    rng = np.random.default_rng(0)
    for _ in range(episodes):
        generator = OrderGenerator(60, base_rate=0.025, horizon=horizon, seed=rng, spike_height=0.0)
        times = [order.arrival_time for order in generator.generate()]
        counts += np.histogram(times, bins=bins, range=(0, horizon))[0]
    reference = OrderGenerator(60, base_rate=0.025, horizon=horizon, seed=0, spike_height=0.0)
    edges = np.linspace(0, horizon, bins + 1)
    expected = np.array([integrate.quad(reference.rate, a, b)[0] for a, b in zip(edges[:-1], edges[1:])]) * episodes
    assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected) + 1)
    assert counts[bins // 2] > 3 * counts[0]  # midday peak vs quiet start


def test_spike_adds_a_rush():
    generator = OrderGenerator(60, base_rate=0.025, seed=3, spike_height=1.5)
    center = generator.spike_center
    assert generator.rate(center) > OrderGenerator(60, base_rate=0.025, seed=3, spike_height=0.0).rate(center)
    assert generator.max_rate >= max(generator.rate(t) for t in np.linspace(0, 720, 2000))


def test_order_fields():
    orders = OrderGenerator(60, seed=1, vip_prob=0.2).generate()
    assert orders and all(0 < o.arrival_time < 720 for o in orders)
    assert [o.arrival_time for o in orders] == sorted(o.arrival_time for o in orders)
    for order in orders:
        assert 1 <= len(order.items) <= 3 and len(set(order.items)) == len(order.items)
        assert all(1 <= w <= 3 for w in order.item_weights) and order.weight == sum(order.item_weights)
        assert order.value > 0 and 0 <= order.dock < 3
        assert not order.vip or order.tier == 0
    tiers = np.bincount([o.tier for o in OrderGenerator(60, seed=2, base_rate=1.0).generate()], minlength=3)
    assert np.allclose(tiers / tiers.sum(), (0.2, 0.6, 0.2), atol=0.04)


def test_hot_shelves_are_zipf_weighted():
    generator = OrderGenerator(60, seed=5, base_rate=2.0)
    picks = np.bincount([item for order in generator.generate() for item in order.items], minlength=60)
    hottest = np.argmax(generator.shelf_probs)
    assert picks[hottest] > 5 * np.median(picks)


def test_truck_schedule():
    trucks = TruckScheduler(dock_count=3, jitter_std=8.0, seed=11).generate(720)
    assert [t.truck_id for t in trucks] == list(range(len(trucks)))
    assert [t.actual_departure for t in trucks] == sorted(t.actual_departure for t in trucks)
    for dock in range(3):
        mine = [t for t in trucks if t.dock == dock]
        assert mine[-1].final and mine[-1].actual_departure == 720
        gaps = np.diff([t.scheduled_departure for t in mine[:-1]])
        assert np.all((gaps >= 120) & (gaps <= 180))
        assert all(abs(t.actual_departure - t.scheduled_departure) <= 24 for t in mine)
