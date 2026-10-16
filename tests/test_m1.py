from warehouse_env.layout import Layout
from warehouse_env.orders import OrderGenerator
from warehouse_env.trucks import TruckScheduler


def test_layout_distance_invariants():
    layout = Layout.from_file("warehouse_env/layouts/default.txt")
    assert layout.distances.dtype.name == "int16"
    assert (layout.distances == layout.distances.T).all()
    assert layout.distance(layout.locations["P"][0], layout.locations["C"][0]) > 0
    for left in layout.positions:
        for middle in layout.positions:
            for right in layout.positions:
                assert layout.distance(left, right) <= layout.distance(left, middle) + layout.distance(middle, right)


def test_generators_are_seed_deterministic():
    first_orders = OrderGenerator(12, seed=7).generate()
    second_orders = OrderGenerator(12, seed=7).generate()
    assert first_orders == second_orders

    first_trucks = TruckScheduler(seed=7).generate()
    second_trucks = TruckScheduler(seed=7).generate()
    assert first_trucks == second_trucks
    assert all(truck.actual_departure >= 0 for truck in first_trucks)