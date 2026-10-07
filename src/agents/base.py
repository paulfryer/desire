"""
Agent interface and shared perception/navigation helpers.

All agents receive the same batched observation dict from NutrientGrid:
    grid   (B, 4, V, V) egocentric one-hot [A, B, C, out-of-bounds], agent at the centre
    intero (B, 3)       internal state h (zeros if interoception is off)
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.env import MOVES, N_TYPES, OOB

STAY = 4


class Agent:
    """Batched agent. `reset(n)` is called before a batch of n episodes that start together."""

    name = "agent"

    def reset(self, n: int) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Option:
    """Go to a visible item by shortest path and eat it (or stay put if item_type is None)."""
    item_type: int | None
    distance: int
    first_action: int


def parse_options(grid: np.ndarray) -> list[Option]:
    """List the reachable items in one egocentric grid (4, V, V) as options, plus 'stay'.

    Paths avoid other items so that approaching one item never consumes another."""
    v = grid.shape[-1]
    centre = (v - 1) // 2
    walls = grid[OOB] > 0.5
    item_type = np.full((v, v), -1, dtype=np.int64)
    for k in range(N_TYPES):
        item_type[grid[k] > 0.5] = k

    dist = np.full((v, v), -1, dtype=np.int64)
    first = np.full((v, v), -1, dtype=np.int64)
    dist[centre, centre] = 0
    queue = deque([(centre, centre)])
    options = [Option(None, 0, STAY)]
    while queue:
        r, c = queue.popleft()
        for a in range(4):
            nr, nc = r + MOVES[a, 0], c + MOVES[a, 1]
            if not (0 <= nr < v and 0 <= nc < v) or walls[nr, nc] or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            first[nr, nc] = a if (r, c) == (centre, centre) else first[r, c]
            if item_type[nr, nc] >= 0:
                options.append(Option(int(item_type[nr, nc]), int(dist[nr, nc]), int(first[nr, nc])))
            else:
                queue.append((nr, nc))
    return options
