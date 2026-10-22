"""Grid layout parsing and all-pairs shortest-path distances.

Legend: ``#`` wall, ``.`` floor, ``S`` shelf (an obstacle picked from an adjacent
floor cell), ``P`` packing/start, ``C`` charger, ``D`` dock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

LAYOUT_DIR = Path(__file__).parent / "layouts"
TRAIN_LAYOUTS = ("train_a", "train_b")
OOD_LAYOUTS = ("ood_a", "ood_b")
_WALKABLE = set(".PCD")


@dataclass(frozen=True, eq=False)
class Layout:
    """Parsed warehouse grid. Robot positions are indices into ``positions``."""

    name: str
    grid: tuple[str, ...]
    positions: tuple[tuple[int, int], ...]
    cell_index: dict[tuple[int, int], int]
    distances: np.ndarray
    shelves: tuple[tuple[int, int], ...]
    shelf_access: np.ndarray
    docks: tuple[int, ...]
    charger: int | None
    packing: int
    blocked: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    @property
    def max_distance(self) -> int:
        return int(self.distances.max())

    @classmethod
    def from_text(cls, text: str, name: str = "custom", blocked: frozenset[tuple[int, int]] = frozenset()) -> Layout:
        rows = tuple(line.rstrip("\n") for line in text.splitlines() if line.strip())
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("layout must contain equally sized, non-empty rows")
        unknown = {cell for row in rows for cell in row} - _WALKABLE - {"#", "S"}
        if unknown:
            raise ValueError(f"unknown layout markers: {sorted(unknown)}")

        walkable = {
            (r, c) for r, row in enumerate(rows) for c, cell in enumerate(row) if cell in _WALKABLE and (r, c) not in blocked
        }
        if not walkable:
            raise ValueError("layout must contain at least one walkable cell")
        positions = tuple(sorted(walkable))
        cell_index = {position: index for index, position in enumerate(positions)}
        distances = _all_pairs_bfs(positions, cell_index)
        if np.any(distances < 0):
            raise ValueError("layout contains disconnected walkable cells")

        def marked(marker: str) -> tuple[int, ...]:
            return tuple(cell_index[p] for p in positions if rows[p[0]][p[1]] == marker)

        shelves = tuple((r, c) for r, row in enumerate(rows) for c, cell in enumerate(row) if cell == "S")
        access = []
        for shelf in shelves:
            neighbors = [n for n in _neighbors(shelf) if n in cell_index]
            if not neighbors:
                raise ValueError(f"shelf at {shelf} has no adjacent floor cell")
            access.append(cell_index[neighbors[0]])
        packing, chargers, docks = marked("P"), marked("C"), marked("D")
        if len(packing) != 1 or len(chargers) > 1 or not docks or len(docks) > 3 or not shelves:
            raise ValueError("layout needs one P, at most one C, 1-3 D and at least one S")
        return cls(
            name=name,
            grid=rows,
            positions=positions,
            cell_index=cell_index,
            distances=distances,
            shelves=shelves,
            shelf_access=np.asarray(access, dtype=np.int64),
            docks=docks,
            charger=chargers[0] if chargers else None,
            packing=packing[0],
            blocked=frozenset(blocked),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Layout:
        path = Path(path)
        return cls.from_text(path.read_text(encoding="utf-8"), name=path.stem)

    @classmethod
    def named(cls, name: str) -> Layout:
        """Load (and cache) a layout shipped in ``warehouse_env/layouts``."""
        return _load_named(name)

    def distance(self, start: int | tuple[int, int], end: int | tuple[int, int]) -> int:
        return int(self.distances[self._index(start), self._index(end)])

    def nearest_distance(self, start: int | tuple[int, int], targets) -> int:
        if not len(targets):
            raise ValueError("targets must not be empty")
        return min(self.distance(start, target) for target in targets)

    def with_blocked_aisles(self, rng: np.random.Generator, count: int = 3, attempts: int = 200) -> Layout:
        """Return a copy with ``count`` floor cells turned into obstacles, keeping every shelf reachable."""
        protected = {self.positions[i] for i in (*self.docks, self.packing)} | {self.positions[i] for i in self.shelf_access}
        if self.charger is not None:
            protected.add(self.positions[self.charger])
        candidates = [p for p in self.positions if p not in protected]
        blocked = set(self.blocked)
        text = "\n".join(self.grid)
        for _ in range(attempts):
            if len(blocked) - len(self.blocked) >= count or not candidates:
                break
            cell = candidates[int(rng.integers(len(candidates)))]
            try:
                Layout.from_text(text, self.name, frozenset(blocked | {cell}))
            except ValueError:
                continue
            blocked.add(cell)
            candidates.remove(cell)
        return Layout.from_text(text, f"{self.name}+blocked", frozenset(blocked))

    def _index(self, cell: int | tuple[int, int]) -> int:
        return cell if isinstance(cell, (int, np.integer)) else self.cell_index[cell]


@cache
def _load_named(name: str) -> Layout:
    return Layout.from_file(LAYOUT_DIR / f"{name}.txt")


def _all_pairs_bfs(positions, cell_index) -> np.ndarray:
    adjacency = [[cell_index[n] for n in _neighbors(p) if n in cell_index] for p in positions]
    distances = np.full((len(positions), len(positions)), -1, dtype=np.int16)
    for start in range(len(positions)):
        row = distances[start]
        row[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if row[neighbor] == -1:
                    row[neighbor] = row[current] + 1
                    queue.append(neighbor)
    return distances


def _neighbors(position: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row, column = position
    return ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
