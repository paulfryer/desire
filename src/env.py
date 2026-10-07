"""
NutrientGrid: a batched homeostatic foraging environment (see docs/RESEARCH_DESIGN_v2.md §3).

The agent lives on a small open grid with items of three types (A, B, C). Eating an item of
type k raises nutrient k of the internal state h. Reward is drive reduction (homeostatic RL),
with drive D(h) = sum_i (h* - h_i)^2, so overshooting a setpoint is punished by the same
function that rewards correcting a deficit.

Observations are egocentric: a (4, 2N-1, 2N-1) grid centred on the agent with channels
[A, B, C, out-of-bounds], which covers the whole N x N grid from any position, plus the
interoceptive vector h (zeros when interoception is off).
"""

from dataclasses import dataclass

import numpy as np

N_TYPES = 3
TYPE_NAMES = ("A", "B", "C")
A, B, C = 0, 1, 2

SETPOINT = 0.5
BITE = 0.3
REWARD_SCALE = 10.0
DEPLETION = np.array([0.004, 0.004, 0.0])  # per step; C never depletes
C_RELAX = 0.05                              # C above setpoint relaxes back toward it
CLAMP_STATE = np.array([0.2, 0.2, SETPOINT])

# up, down, left, right, stay (row, col deltas)
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
N_ACTIONS = len(MOVES)
OOB = 3  # out-of-bounds channel index


def drive(h: np.ndarray) -> np.ndarray:
    """Homeostatic drive D(h) = sum_i (h* - h_i)^2 over the last axis."""
    return ((SETPOINT - h) ** 2).sum(axis=-1)


def body_drift(h: np.ndarray) -> np.ndarray:
    """One step of the body's own dynamics (no eating): A and B deplete, C above setpoint relaxes.
    Matches NutrientGrid.step for free need dynamics; planners are given this as innate."""
    h = h - DEPLETION
    c = h[..., C]
    h[..., C] = np.where(c > SETPOINT, c + C_RELAX * (SETPOINT - c), c)
    return np.clip(h, 0.0, 1.0)


def infer_increment(h_before: np.ndarray, h_after: np.ndarray) -> np.ndarray:
    """Invert body_drift to recover what eating did to each nutrient this step.

    Returns the increment per nutrient, or NaN where it cannot be recovered (the nutrient hit
    the ceiling of 1 or the floor of 0, which censors the increment)."""
    x = h_after + DEPLETION
    c = h_after[..., C]
    x[..., C] = np.where(c > SETPOINT, (c - C_RELAX * SETPOINT) / (1 - C_RELAX), c)
    delta = x - h_before
    censored = (x >= 1.0 - 1e-6) | (h_after <= 1e-6)
    return np.where(censored, np.nan, delta)


@dataclass(frozen=True)
class EnvConfig:
    grid: int = 9
    need_dynamics: str = "free"   # "free": needs vary and deplete; "clamped": h reset every step
    interoception: bool = True    # observe h, or zeros
    episode_steps: int = 200
    probe_steps: int = 25
    # "reduction": r = 10 (D(h_t) - D(h_t+1)); "penalty": r = -D(h_t+1). With discount gamma the
    # two have the same optimal policies (the reduction return is D(h_0) - (1-gamma)/gamma times
    # the penalty return), but the penalty form is far better conditioned for policy gradients.
    reward: str = "reduction"

    def __post_init__(self):
        if self.need_dynamics not in ("free", "clamped"):
            raise ValueError(f"need_dynamics must be 'free' or 'clamped', got {self.need_dynamics!r}")
        if self.reward not in ("reduction", "penalty"):
            raise ValueError(f"reward must be 'reduction' or 'penalty', got {self.reward!r}")

    @property
    def view(self) -> int:
        return 2 * self.grid - 1


@dataclass
class ProbeSpec:
    """A batch of choice probes: one agent position, two items and an internal state each."""
    agent_pos: np.ndarray    # (B, 2)
    item_pos: np.ndarray     # (B, 2, 2)
    item_types: np.ndarray   # (B, 2)
    h: np.ndarray            # (B, 3)


class NutrientGrid:
    """Batch of independent NutrientGrid episodes, stepped together."""

    def __init__(self, n_envs: int, config: EnvConfig = EnvConfig(), seed: int | None = None):
        self.n = n_envs
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        g = config.grid
        self.pos = np.zeros((n_envs, 2), dtype=np.int64)
        self.items = np.zeros((n_envs, N_TYPES, g, g), dtype=bool)
        self.h = np.zeros((n_envs, N_TYPES))
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.probe = False
        self._env_idx = np.arange(n_envs)
        # Padded allocentric canvas for egocentric crops; OOB channel is 1 outside the grid.
        pad = g - 1
        self._pad = pad
        self._canvas = np.zeros((n_envs, N_TYPES + 1, g + 2 * pad, g + 2 * pad), dtype=np.float32)
        self._canvas[:, OOB] = 1.0
        self._canvas[:, OOB, pad:pad + g, pad:pad + g] = 0.0
        self._view_offsets = np.arange(config.view)

    # ------------------------------------------------------------------ resets

    def reset(self, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Start fresh training episodes for envs in mask (all if None); return all observations."""
        self.probe = False
        idx = self._env_idx if mask is None else np.flatnonzero(mask)
        g = self.cfg.grid
        for b in idx:
            counts = self.rng.integers(0, 3, size=N_TYPES)
            while counts.sum() < 2:
                counts = self.rng.integers(0, 3, size=N_TYPES)
            cells = self.rng.choice(g * g, size=1 + counts.sum(), replace=False)
            self.pos[b] = divmod(cells[0], g)
            self.items[b] = False
            types = np.repeat(np.arange(N_TYPES), counts)
            for cell, k in zip(cells[1:], types, strict=True):
                self.items[b, k, cell // g, cell % g] = True
            if self.cfg.need_dynamics == "free":
                self.h[b] = [self.rng.uniform(0.05, 0.8), self.rng.uniform(0.05, 0.8), SETPOINT]
            else:
                self.h[b] = CLAMP_STATE
            self.t[b] = 0
        return self.observe()

    def reset_probe(self, spec: ProbeSpec) -> dict[str, np.ndarray]:
        """Set up choice probes (requires len(spec) == n_envs). Probes end at the first item eaten."""
        if len(spec.agent_pos) != self.n:
            raise ValueError("probe batch size must equal n_envs")
        self.probe = True
        self.pos[:] = spec.agent_pos
        self.items[:] = False
        for slot in range(2):
            self.items[self._env_idx, spec.item_types[:, slot],
                       spec.item_pos[:, slot, 0], spec.item_pos[:, slot, 1]] = True
        self.h[:] = spec.h
        self.t[:] = 0
        return self.observe()

    # ------------------------------------------------------------------ dynamics

    def step(self, actions: np.ndarray):
        """Advance every env one step. Returns (obs, reward, done, eaten) where eaten is the
        type consumed this step (-1 if none). Done envs are not reset automatically."""
        actions = np.asarray(actions, dtype=np.int64)
        g = self.cfg.grid
        self.pos = np.clip(self.pos + MOVES[actions], 0, g - 1)
        rows, cols = self.pos[:, 0], self.pos[:, 1]

        here = self.items[self._env_idx, :, rows, cols]          # (B, 3)
        ate = here.any(axis=1)
        eaten = np.where(ate, here.argmax(axis=1), -1)

        d_before = drive(self.h)
        eat_idx = np.flatnonzero(ate)
        if eat_idx.size:
            k = eaten[eat_idx]
            self.h[eat_idx, k] = np.minimum(1.0, self.h[eat_idx, k] + BITE)
            self.items[eat_idx, k, rows[eat_idx], cols[eat_idx]] = False
            if not self.probe:
                for b, kb in zip(eat_idx, k, strict=True):
                    self._respawn(b, kb)

        self.h -= DEPLETION
        above = self.h[:, C] > SETPOINT
        self.h[above, C] += C_RELAX * (SETPOINT - self.h[above, C])
        np.clip(self.h, 0.0, 1.0, out=self.h)
        if self.cfg.reward == "reduction":
            reward = REWARD_SCALE * (d_before - drive(self.h))
        else:
            reward = -drive(self.h)

        if self.cfg.need_dynamics == "clamped" and not self.probe:
            self.h[:] = CLAMP_STATE

        self.t += 1
        if self.probe:
            done = ate | (self.t >= self.cfg.probe_steps)
        else:
            done = self.t >= self.cfg.episode_steps
        return self.observe(), reward.astype(np.float32), done, eaten

    def _respawn(self, b: int, k: int) -> None:
        g = self.cfg.grid
        free = ~self.items[b].any(axis=0)
        free[self.pos[b, 0], self.pos[b, 1]] = False
        cells = np.flatnonzero(free)
        cell = self.rng.choice(cells)
        self.items[b, k, cell // g, cell % g] = True

    # ------------------------------------------------------------------ observations

    def observe(self) -> dict[str, np.ndarray]:
        g, pad, v = self.cfg.grid, self._pad, self.cfg.view
        self._canvas[:, :N_TYPES, pad:pad + g, pad:pad + g] = self.items
        # The egocentric window for an agent at (r, c) starts at canvas (r, c) because pad = g - 1.
        rows = self.pos[:, 0, None] + self._view_offsets              # (B, v)
        cols = self.pos[:, 1, None] + self._view_offsets              # (B, v)
        crop = self._canvas[self._env_idx[:, None, None, None],
                            np.arange(N_TYPES + 1)[None, :, None, None],
                            rows[:, None, :, None],
                            cols[:, None, None, :]]                   # (B, 4, v, v)
        intero = self.h.astype(np.float32) if self.cfg.interoception else \
            np.zeros((self.n, N_TYPES), dtype=np.float32)
        assert crop.shape == (self.n, N_TYPES + 1, v, v)
        return {"grid": crop, "intero": intero.copy()}

    def mean_drive(self) -> float:
        return float(drive(self.h).mean())


def sample_probe_positions(rng: np.random.Generator, n: int, grid: int = 9,
                           distances=(3, 4, 5)) -> tuple[np.ndarray, np.ndarray]:
    """Sample agent positions and two item cells at equal Manhattan distance whose
    distance-reducing first moves are disjoint (so the first move reveals the choice).
    Returns agent_pos (n, 2) and item_pos (n, 2, 2)."""
    agent = np.zeros((n, 2), dtype=np.int64)
    items = np.zeros((n, 2, 2), dtype=np.int64)
    coords = np.array([(r, c) for r in range(grid) for c in range(grid)])
    i = 0
    while i < n:
        a = coords[rng.integers(len(coords))]
        d = rng.choice(distances)
        ring = coords[np.abs(coords - a).sum(axis=1) == d]
        if len(ring) < 2:
            continue
        p, q = ring[rng.choice(len(ring), size=2, replace=False)]
        if _first_moves(a, p) & _first_moves(a, q):
            continue
        agent[i], items[i, 0], items[i, 1] = a, p, q
        i += 1
    return agent, items


def _first_moves(a: np.ndarray, target: np.ndarray) -> set[int]:
    """Actions that reduce Manhattan distance from a to target."""
    moves = set()
    dr, dc = target - a
    if dr < 0:
        moves.add(0)
    if dr > 0:
        moves.add(1)
    if dc < 0:
        moves.add(2)
    if dc > 0:
        moves.add(3)
    return moves
