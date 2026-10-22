import numpy as np
import pytest

from warehouse_env.layout import LAYOUT_DIR, Layout

LAYOUT_NAMES = sorted(path.stem for path in LAYOUT_DIR.glob("*.txt"))


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_distance_matrix_is_a_metric(name):
    layout = Layout.named(name)
    d = layout.distances.astype(np.int32)
    assert layout.distances.dtype == np.int16
    assert (d == d.T).all()
    assert (np.diag(d) == 0).all()
    assert (d[~np.eye(len(d), dtype=bool)] > 0).all()
    for middle in range(len(d)):  # triangle inequality through every intermediate cell
        assert (d <= d[:, [middle]] + d[[middle], :]).all()


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_shelves_have_adjacent_access_cells(name):
    layout = Layout.named(name)
    for shelf, access in zip(layout.shelves, layout.shelf_access):
        row, col = layout.positions[access]
        assert abs(row - shelf[0]) + abs(col - shelf[1]) == 1
        assert layout.grid[shelf[0]][shelf[1]] == "S"


def test_full_layouts_match_plan_scale():
    for name in ("train_a", "train_b", "ood_a", "ood_b"):
        layout = Layout.named(name)
        assert layout.shape == (15, 21)
        assert 50 <= len(layout.shelves) <= 75
        assert len(layout.docks) == 3 and layout.charger is not None


def test_blocked_aisles_keep_layout_connected():
    base = Layout.named("train_a")
    blocked = base.with_blocked_aisles(np.random.default_rng(0), count=5)
    assert len(blocked.blocked) == 5
    assert len(blocked.positions) == len(base.positions) - 5
    assert (blocked.distances >= 0).all()
    assert blocked.distances.mean() >= base.distances.mean() - 1e-9


def test_invalid_layouts_are_rejected():
    with pytest.raises(ValueError):
        Layout.from_text("###\n#P#\n##")
    with pytest.raises(ValueError):
        Layout.from_text("#####\n#P#D#\n#####")  # disconnected
    with pytest.raises(ValueError):
        Layout.from_text("####\n#PX#\n####")
