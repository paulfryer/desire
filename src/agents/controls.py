"""
Ground-truth control agents (no learning). Their desire status is known by construction, which
is what lets them validate the assays (docs/RESEARCH_DESIGN_v2.md §4, known-groups validity).
"""

import numpy as np

from src.agents.base import Agent, parse_options
from src.env import N_ACTIONS, SETPOINT, A, B, C


class RandomAgent(Agent):
    """Uniform random actions: no desire."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return self.rng.integers(0, N_ACTIONS, size=len(obs["intero"]))


class _TypePreferenceAgent(Agent):
    """Goes to the nearest item among a fixed ordered set of acceptable types, ignoring needs."""

    tiers: tuple[tuple[int, ...], ...] = ()

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        actions = np.empty(len(obs["intero"]), dtype=np.int64)
        for b, grid in enumerate(obs["grid"]):
            actions[b] = self._act_one(parse_options(grid))
        return actions

    def _act_one(self, options):
        for tier in self.tiers:
            candidates = [o for o in options if o.item_type in tier]
            if candidates:
                best = min(o.distance for o in candidates)
                nearest = [o for o in candidates if o.distance == best]
                return nearest[self.rng.integers(len(nearest))].first_action
        return int(self.rng.integers(0, 4))


class NearestAgent(_TypePreferenceAgent):
    """Nearest A or B item, never C, regardless of need: an outcome-general habit."""

    name = "nearest"
    tiers = ((A, B),)


class FixedHabitAgent(_TypePreferenceAgent):
    """Always A if any is present, else B, regardless of need: an outcome-specific S-R habit."""

    name = "fixed_habit"
    tiers = ((A,), (B,))


class NeedNoveltyAgent(_TypePreferenceAgent):
    """Like `nearest`, but whenever any need is strong it heads for C, the item it has never
    needed. It has no knowledge of what C does: a T2 'pass' by novelty-seeking under need, which
    the specificity guard (RI_spec) must catch."""

    name = "need_novelty"
    strong_need = 0.3

    def act(self, obs):
        actions = np.empty(len(obs["intero"]), dtype=np.int64)
        for b, grid in enumerate(obs["grid"]):
            needy = (SETPOINT - obs["intero"][b]).max() > self.strong_need
            self.tiers = ((C,), (A, B)) if needy else ((A, B),)
            actions[b] = self._act_one(parse_options(grid))
        return actions
