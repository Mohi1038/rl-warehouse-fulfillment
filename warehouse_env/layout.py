"""Grid layout parsing and all-pairs shortest-path distances."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Layout:
    """Parsed warehouse grid with a compact index for walkable cells."""

    grid: tuple[str, ...]
    cell_index: dict[tuple[int, int], int]
    positions: tuple[tuple[int, int], ...]
    locations: dict[str, tuple[tuple[int, int], ...]]
    distances: np.ndarray

    @classmethod
    def from_text(cls, text: str) -> "Layout":
        rows = tuple(line.rstrip("\n") for line in text.splitlines() if line.strip())
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("layout must contain equally sized, non-empty rows")

        walkable = {(r, c) for r, row in enumerate(rows) for c, cell in enumerate(row) if cell != "#"}
        if not walkable:
            raise ValueError("layout must contain at least one walkable cell")
        positions = tuple(sorted(walkable))
        cell_index = {position: index for index, position in enumerate(positions)}
        locations: dict[str, list[tuple[int, int]]] = {}
        for position in positions:
            marker = rows[position[0]][position[1]]
            if marker != ".":
                locations.setdefault(marker, []).append(position)

        distances = np.full((len(positions), len(positions)), -1, dtype=np.int16)
        for start in positions:
            start_index = cell_index[start]
            distances[start_index, start_index] = 0
            queue = deque([start])
            while queue:
                current = queue.popleft()
                current_distance = int(distances[start_index, cell_index[current]])
                for neighbor in _neighbors(current):
                    if neighbor in cell_index:
                        neighbor_index = cell_index[neighbor]
                        if distances[start_index, neighbor_index] == -1:
                            distances[start_index, neighbor_index] = current_distance + 1
                            queue.append(neighbor)

        if np.any(distances < 0):
            raise ValueError("layout contains disconnected walkable cells")
        return cls(rows, cell_index, positions, {key: tuple(value) for key, value in locations.items()}, distances)

    @classmethod
    def from_file(cls, path: str | Path) -> "Layout":
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def distance(self, start: tuple[int, int], end: tuple[int, int]) -> int:
        return int(self.distances[self.cell_index[start], self.cell_index[end]])

    def nearest_distance(self, start: tuple[int, int], targets: tuple[tuple[int, int], ...]) -> int:
        if not targets:
            raise ValueError("targets must not be empty")
        return min(self.distance(start, target) for target in targets)


def _neighbors(position: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row, column = position
    return ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))