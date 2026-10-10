"""
MultiNutrientGrid: the held-out-need environment (docs/HELDOUT_DESIGN.md §3).

Same mechanics as NutrientGrid (src/env.py), with N = 7 nutrients, each identified by a code:
one row of a random 8 x 8 orthogonal matrix (same geometry in every seed). An item displays the
code of the nutrient it restores (a labeled line).

- K nutrients *deplete* (the trained needs). Total depletion is matched across K.
- Nutrient H = N - 1 is held out: it never depletes. Each episode it starts at U[0.5, 0.8] (or
  exactly 0.5 if held_out_constant) and relaxes back toward the set point, so the agent
  experiences H's items and H's full side, but never an H deficit.
- Any other non-depleting ("dormant") nutrients behave like H.

Observations are object-based:
- items: (2N, CODE_DIM + 3) tokens [code, d_row / 8, d_col / 8, present], nearest first, ties random;
- needs: (N, CODE_DIM + 1) tokens [code, level], sorted by level (lowest first), ties random;
- pos: the agent's (row, col) / 8.
"""

from dataclasses import dataclass

import numpy as np

from src.env import BITE, MOVES, REWARD_SCALE, SETPOINT, ProbeSpec, drive

N_NUTRIENTS = 7
CODE_DIM = 8
TOTAL_DEPLETION = 0.024     # per step, summed over depleting nutrients (6 x 0.004)
RELAX = 0.05                # non-depleting nutrients above the set point relax back toward it
FULL_SIDE = (0.5, 0.8)      # initial range for non-depleting nutrients
SPARSE_EPISODES = 0.25      # share of episodes with exactly two items (like probe displays)


@dataclass(frozen=True)
class World:
    """Everything fixed for one training seed."""
    n: int
    n_depleting: int
    held_out_constant: bool
    codes: np.ndarray           # (n, CODE_DIM) orthonormal rows; items and needs share them
    spare_code: np.ndarray      # (CODE_DIM,) orthogonal to every nutrient code (novel-code probe)

    @property
    def held_out(self) -> int:
        return self.n - 1

    @property
    def trained(self) -> list[int]:
        return list(range(self.n_depleting))

    @property
    def depletion(self) -> np.ndarray:
        d = np.zeros(self.n)
        d[: self.n_depleting] = TOTAL_DEPLETION / self.n_depleting
        return d

    def with_held_out_code(self, code: np.ndarray) -> "World":
        """The same world with H's code replaced (used only by the novel-code probe)."""
        codes = self.codes.copy()
        codes[self.held_out] = code
        return World(self.n, self.n_depleting, self.held_out_constant, codes, self.spare_code)


def make_world(seed: int, n_depleting: int = 6, held_out_constant: bool = False,
               n: int = N_NUTRIENTS) -> World:
    """Codes depend only on the seed, so arms trained with the same seed share them."""
    rng = np.random.default_rng([seed, 2_718])
    q, r = np.linalg.qr(rng.normal(size=(CODE_DIM, CODE_DIM)))
    q = q * np.sign(np.diag(r))          # a uniformly random orthogonal matrix
    return World(n, n_depleting, held_out_constant, q[:n].copy(), q[n].copy())


def body_drift(h: np.ndarray, world: World) -> np.ndarray:
    """One step of the body's own dynamics (no eating), matching MultiNutrientGrid.step."""
    dep = world.depletion
    h = h - dep
    relax = (dep == 0) & (h > SETPOINT)
    h = np.where(relax, h + RELAX * (SETPOINT - h), h)
    return np.clip(h, 0.0, 1.0)


@dataclass(frozen=True)
class MultiConfig:
    grid: int = 9
    episode_steps: int = 200
    probe_steps: int = 25


class MultiNutrientGrid:
    """Batch of independent episodes in one world, stepped together."""

    def __init__(self, n_envs: int, world: World, seed: int | None = None,
                 config: MultiConfig = MultiConfig()):
        self.n_envs = n_envs
        self.world = world
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        g, n = config.grid, world.n
        self.max_items = 2 * n
        self.pos = np.zeros((n_envs, 2), dtype=np.int64)
        self.items = np.zeros((n_envs, n, g, g), dtype=bool)
        self.h = np.zeros((n_envs, n))
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.probe = False
        self._idx = np.arange(n_envs)
        self._depletion = world.depletion
        self._relaxing = self._depletion == 0

    # ------------------------------------------------------------------ resets

    def _initial_levels(self) -> np.ndarray:
        n = self.world.n
        h = np.where(self._relaxing, self.rng.uniform(*FULL_SIDE, size=n),
                     self.rng.uniform(0.05, 0.8, size=n))
        if self.world.held_out_constant:
            h[self.world.held_out] = SETPOINT
        return h

    def reset(self, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        self.probe = False
        idx = self._idx if mask is None else np.flatnonzero(mask)
        g, n = self.cfg.grid, self.world.n
        for b in idx:
            if self.rng.random() < SPARSE_EPISODES:
                counts = np.zeros(n, dtype=np.int64)
                counts[self.rng.choice(n, size=2, replace=False)] = 1
            else:
                counts = self.rng.integers(0, 3, size=n)
                while counts.sum() < 2:
                    counts = self.rng.integers(0, 3, size=n)
            cells = self.rng.choice(g * g, size=1 + counts.sum(), replace=False)
            self.pos[b] = divmod(cells[0], g)
            self.items[b] = False
            for cell, k in zip(cells[1:], np.repeat(np.arange(n), counts), strict=True):
                self.items[b, k, cell // g, cell % g] = True
            self.h[b] = self._initial_levels()
            self.t[b] = 0
        return self.observe()

    def reset_probe(self, spec: ProbeSpec) -> dict[str, np.ndarray]:
        if len(spec.agent_pos) != self.n_envs:
            raise ValueError("probe batch size must equal n_envs")
        self.probe = True
        self.pos[:] = spec.agent_pos
        self.items[:] = False
        for slot in range(2):
            self.items[self._idx, spec.item_types[:, slot],
                       spec.item_pos[:, slot, 0], spec.item_pos[:, slot, 1]] = True
        self.h[:] = spec.h
        self.t[:] = 0
        return self.observe()

    # ------------------------------------------------------------------ dynamics

    def step(self, actions: np.ndarray):
        """Returns (obs, reward, done, eaten); eaten is the type consumed (-1 if none)."""
        actions = np.asarray(actions, dtype=np.int64)
        g = self.cfg.grid
        self.pos = np.clip(self.pos + MOVES[actions], 0, g - 1)
        rows, cols = self.pos[:, 0], self.pos[:, 1]
        here = self.items[self._idx, :, rows, cols]
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

        self.h -= self._depletion
        relax = self._relaxing & (self.h > SETPOINT)
        self.h[relax] += RELAX * (SETPOINT - self.h[relax])
        np.clip(self.h, 0.0, 1.0, out=self.h)
        reward = REWARD_SCALE * (d_before - drive(self.h))

        self.t += 1
        done = (ate | (self.t >= self.cfg.probe_steps)) if self.probe else \
            self.t >= self.cfg.episode_steps
        return self.observe(), reward.astype(np.float32), done, eaten

    def _respawn(self, b: int, k: int) -> None:
        g = self.cfg.grid
        free = ~self.items[b].any(axis=0)
        free[self.pos[b, 0], self.pos[b, 1]] = False
        cell = self.rng.choice(np.flatnonzero(free))
        self.items[b, k, cell // g, cell % g] = True

    # ------------------------------------------------------------------ observations

    def observe(self) -> dict[str, np.ndarray]:
        w, g = self.world, self.cfg.grid
        scale = float(g - 1)
        b, k, r, c = np.nonzero(self.items)
        dr, dc = r - self.pos[b, 0], c - self.pos[b, 1]
        dist = np.abs(dr) + np.abs(dc)
        order = np.lexsort((self.rng.random(b.size), dist, b))   # by env, then distance, ties random
        b, k, dr, dc = b[order], k[order], dr[order], dc[order]
        rank = np.arange(b.size) - np.searchsorted(b, b, side="left")
        keep = rank < self.max_items
        b, k, dr, dc, rank = b[keep], k[keep], dr[keep], dc[keep], rank[keep]
        items = np.zeros((self.n_envs, self.max_items, CODE_DIM + 3), dtype=np.float32)
        items[b, rank, :CODE_DIM] = w.codes[k]
        items[b, rank, CODE_DIM] = dr / scale
        items[b, rank, CODE_DIM + 1] = dc / scale
        items[b, rank, CODE_DIM + 2] = 1.0

        jitter = self.rng.random((self.n_envs, w.n)) * 1e-6
        perm = np.argsort(self.h + jitter, axis=1)                  # lowest level first, ties random
        needs = np.empty((self.n_envs, w.n, CODE_DIM + 1), dtype=np.float32)
        needs[..., :CODE_DIM] = w.codes[perm]
        needs[..., CODE_DIM] = np.take_along_axis(self.h, perm, axis=1)
        return {"items": items, "needs": needs, "pos": (self.pos / scale).astype(np.float32)}

    def mean_drive(self) -> float:
        return float(drive(self.h).mean())
