"""
Agents for the held-out-need world (docs/HELDOUT_DESIGN.md): ground-truth controls, the
stable-baselines3 VecEnv adapter, and the two learned architectures (token MLP, relational).
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from torch import nn

from src.agents.base import STAY, Agent
from src.env import N_ACTIONS, SETPOINT, drive
from src.env_n import (
    BITE,
    CODE_DIM,
    FULL_SIDE,
    MultiConfig,
    MultiNutrientGrid,
    World,
    body_drift,
)

# ---------------------------------------------------------------------- decoding observations


def _decode(codes: np.ndarray, world: World) -> np.ndarray:
    """Nutrient index for each code (the spare code used by the novel-code probe reads as H)."""
    table = np.vstack([world.codes, world.spare_code])
    idx = np.argmax(codes @ table.T, axis=-1)
    return np.where(idx == world.n, world.held_out, idx)


def item_tokens(obs: dict, world: World, grid: int = 9):
    """Per env: list of (type, d_row, d_col) for present items, decoded from item codes."""
    scale = grid - 1
    out = []
    for tokens in obs["items"]:
        present = tokens[:, CODE_DIM + 2] > 0.5
        t = tokens[present]
        types = _decode(t[:, :CODE_DIM], world)
        dr = np.rint(t[:, CODE_DIM] * scale).astype(int)
        dc = np.rint(t[:, CODE_DIM + 1] * scale).astype(int)
        out.append(list(zip(types.tolist(), dr.tolist(), dc.tolist(), strict=True)))
    return out


def need_levels(obs: dict, world: World) -> np.ndarray:
    """(B, N) need levels in nutrient order, decoded from the shuffled need tokens."""
    which = _decode(obs["needs"][..., :CODE_DIM], world)       # (B, N) nutrient of each token
    levels = np.zeros(which.shape)
    np.put_along_axis(levels, which, obs["needs"][..., CODE_DIM], axis=1)
    return levels


def step_toward(target: tuple[int, int], others: list[tuple[int, int]],
                rng: np.random.Generator) -> int:
    """A move that reduces Manhattan distance to target without stepping onto another item."""
    dr, dc = target
    moves = []
    if dr < 0:
        moves.append((0, (-1, 0)))
    if dr > 0:
        moves.append((1, (1, 0)))
    if dc < 0:
        moves.append((2, (0, -1)))
    if dc > 0:
        moves.append((3, (0, 1)))
    safe = [a for a, d in moves if d not in others or d == target]
    pool = safe or [a for a, _ in moves]
    return int(rng.choice(pool)) if pool else STAY


# ---------------------------------------------------------------------- controls


class _TargetAgent(Agent):
    """Picks a target item each step (subclasses define how) and walks toward it."""

    def __init__(self, world: World, seed: int | None = None):
        self.world = world
        self.rng = np.random.default_rng(seed)

    def reset(self, n):
        pass

    def target(self, items, levels):  # -> index into items, or None to stay
        raise NotImplementedError

    def act(self, obs):
        levels = need_levels(obs, self.world)
        actions = np.full(len(levels), STAY)
        for b, items in enumerate(item_tokens(obs, self.world)):
            if not items:
                continue
            i = self.target(items, levels[b])
            if i is None:
                continue
            others = [(dr, dc) for j, (_, dr, dc) in enumerate(items) if j != i]
            actions[b] = step_toward(items[i][1:], others, self.rng)
        return actions

    def _nearest(self, items, allowed=None):
        idx = [i for i, it in enumerate(items) if allowed is None or it[0] in allowed]
        if not idx:
            return None
        dist = np.array([abs(items[i][1]) + abs(items[i][2]) for i in idx])
        best = np.flatnonzero(dist == dist.min())
        return idx[int(self.rng.choice(best))]


class MultiOracle(_TargetAgent):
    """Ideal wanting: true item effects and the drive function, planned over a short horizon."""

    name = "oracle"

    def __init__(self, world: World, seed: int | None = None, horizon: int = 20):
        super().__init__(world, seed)
        self.horizon = horizon

    def _score(self, h, item_type, distance):
        total = 0.0
        for t in range(1, self.horizon + 1):
            if item_type is not None and t == distance:
                h = h.copy()
                h[item_type] = min(1.0, h[item_type] + BITE)
            h = body_drift(h, self.world)
            total += drive(h)
        return total

    def target(self, items, levels):
        scores = [self._score(levels, k, abs(dr) + abs(dc)) for k, dr, dc in items]
        stay = self._score(levels, None, 0)
        best = np.flatnonzero(np.isclose(scores, min(scores)))
        if min(scores) >= stay:
            return None
        return int(self.rng.choice(best))


class MultiNearest(_TargetAgent):
    """Habit: always the nearest item, whatever it is."""

    name = "nearest"

    def target(self, items, levels):
        return self._nearest(items)


class MultiNeedNovelty(_TargetAgent):
    """No desire, just novelty under need: seeks H items whenever any deficit exceeds 0.3;
    otherwise prefers familiar (trained) items. Must pass RI_H and fail the guard."""

    name = "need_novelty"

    def target(self, items, levels):
        needy = (SETPOINT - levels).max() > 0.3
        prefer = {self.world.held_out} if needy else set(self.world.trained)
        i = self._nearest(items, prefer)
        return i if i is not None else self._nearest(items)


class MultiVeto(_TargetAgent):
    """Impostor: only avoids items whose nutrient is at or above the set point, otherwise picks
    the nearest. Lifts a satiety veto but never prefers the needier item (RI 0.5, SI 0)."""

    name = "veto"

    def target(self, items, levels):
        return self._nearest(items, {k for k, _, _ in items if levels[k] < SETPOINT})


class MultiBreakdown(MultiOracle):
    """Impostor: ideal on familiar inputs, but chooses at random whenever H's level is outside
    the range H ever had in training (its full side)."""

    name = "breakdown"

    def target(self, items, levels):
        h = levels[self.world.held_out]
        if not FULL_SIDE[0] <= h <= FULL_SIDE[1]:
            return int(self.rng.integers(len(items)))
        return super().target(items, levels)


class MultiUnusualSeeker(MultiOracle):
    """Impostor: goes for H's item whenever H's level is unusual for H, in either direction,
    otherwise ideal. Passes RI_H, RI_spec_H and SI_H; only RI_dir_H catches it."""

    name = "unusual_seeker"

    def target(self, items, levels):
        h = levels[self.world.held_out]
        if not FULL_SIDE[0] <= h <= FULL_SIDE[1]:
            i = self._nearest(items, {self.world.held_out})
            if i is not None:
                return i
        return super().target(items, levels)


class MultiRandom(Agent):
    name = "random"

    def __init__(self, world: World | None = None, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, n):
        pass

    def act(self, obs):
        return self.rng.integers(0, N_ACTIONS, size=len(obs["pos"]))


CONTROLS = {"oracle": MultiOracle, "nearest": MultiNearest, "need_novelty": MultiNeedNovelty,
            "veto": MultiVeto, "breakdown": MultiBreakdown, "unusual_seeker": MultiUnusualSeeker,
            "random": MultiRandom}


# ---------------------------------------------------------------------- SB3 adapter


def observation_space(world: World) -> spaces.Dict:
    n = world.n
    return spaces.Dict({
        "items": spaces.Box(-1.0, 1.0, shape=(2 * n, CODE_DIM + 3), dtype=np.float32),
        "needs": spaces.Box(-1.0, 1.0, shape=(n, CODE_DIM + 1), dtype=np.float32),
        "pos": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
    })


class MultiVecEnv(VecEnv):
    """stable-baselines3 VecEnv over MultiNutrientGrid (auto-resets finished episodes)."""

    def __init__(self, n_envs: int, world: World, seed: int | None = None,
                 config: MultiConfig = MultiConfig()):
        self.grid_env = MultiNutrientGrid(n_envs, world, seed, config)
        self.grid_env.render_mode = None
        super().__init__(n_envs, observation_space(world), spaces.Discrete(N_ACTIONS))
        self._actions = np.zeros(n_envs, dtype=np.int64)
        self._returns = np.zeros(n_envs)
        self._lengths = np.zeros(n_envs, dtype=np.int64)
        self._eaten = np.zeros((n_envs, world.n), dtype=np.int64)

    def reset(self):
        self._returns[:] = 0
        self._lengths[:] = 0
        self._eaten[:] = 0
        return self.grid_env.reset()

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        obs, reward, done, eaten = self.grid_env.step(self._actions)
        self._returns += reward
        self._lengths += 1
        ate = eaten >= 0
        self._eaten[np.flatnonzero(ate), eaten[ate]] += 1
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            for b in np.flatnonzero(done):
                infos[b]["terminal_observation"] = {k: v[b].copy() for k, v in obs.items()}
                infos[b]["TimeLimit.truncated"] = True
                infos[b]["episode"] = {"r": float(self._returns[b]), "l": int(self._lengths[b]),
                                       "eaten": self._eaten[b].copy()}
            self._returns[done] = 0
            self._lengths[done] = 0
            self._eaten[done] = 0
            obs = self.grid_env.reset(done)
        return obs, reward, done.copy(), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.grid_env, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.grid_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        result = getattr(self.grid_env, method_name)(*method_args, **method_kwargs)
        return [result] * len(self._indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def seed(self, seed=None):
        self.grid_env.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices


# ---------------------------------------------------------------------- learned architectures


class TokenMLP(BaseFeaturesExtractor):
    """Generic learner: every token flattened and concatenated; the policy and value MLPs on
    top must discover for themselves which item restores which need."""

    def __init__(self, observation_space: spaces.Dict):
        dim = sum(int(np.prod(observation_space[k].shape)) for k in ("items", "needs", "pos"))
        super().__init__(observation_space, features_dim=dim)

    def forward(self, obs):
        return torch.cat([obs["items"].flatten(1), obs["needs"].flatten(1), obs["pos"]], dim=1)


class RelationalTokens(BaseFeaturesExtractor):
    """Relational learner. Each item attends over the need tokens by content:
    - binding="innate": attention logits are tau * (item code . need code), a built-in labeled
      line (tau learned);
    - binding="learned": logits come from learned, untied W_q and W_k, so even the match must
      be learned.
    It reads out a learned embedding of the attended needs: of the level only (readout="level")
    or of code and level ("code"). The read-out is combined with the item's own embedding (code
    and position), so code-specific habits such as "avoid H items" are always representable.
    Items are pooled by learned attention for the policy and value heads."""

    def __init__(self, observation_space: spaces.Dict, width: int = 64, readout: str = "level",
                 binding: str = "innate"):
        super().__init__(observation_space, features_dim=2 * width + 2)
        d = CODE_DIM
        self.readout, self.binding = readout, binding
        if binding == "learned":
            self.q = nn.Linear(d, width, bias=False)
            self.k = nn.Linear(d, width, bias=False)
        else:
            self.tau = nn.Parameter(torch.tensor(3.0))
        n_in = 1 if readout == "level" else d + 1
        self.need_value = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width))
        self.item_embed = nn.Sequential(nn.Linear(d + 3, width), nn.ReLU())
        self.combine = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU())
        self.pool_score = nn.Linear(width, 1)
        self.need_summary = nn.Sequential(nn.Linear(d + 1, width), nn.ReLU())
        self.width = width

    def forward(self, obs):
        items, needs = obs["items"], obs["needs"]
        present = items[..., -1] > 0.5                                     # (B, M)
        if self.binding == "learned":
            q = self.q(items[..., :CODE_DIM])                              # (B, M, w)
            k = self.k(needs[..., :CODE_DIM])                              # (B, N, w)
            logits = q @ k.transpose(1, 2) / self.width ** 0.5
        else:
            logits = self.tau * (items[..., :CODE_DIM] @ needs[..., :CODE_DIM].transpose(1, 2))
        att = torch.softmax(logits, dim=-1)                                # (B, M, N)
        value_in = needs[..., CODE_DIM:] if self.readout == "level" else needs
        read = att @ self.need_value(value_in)                             # (B, M, w)
        f = self.combine(torch.cat([self.item_embed(items), read], dim=-1))  # (B, M, w)
        score = self.pool_score(f).squeeze(-1).masked_fill(~present, -1e9)
        any_item = present.any(dim=1, keepdim=True)
        weights = torch.softmax(score, dim=1) * any_item
        pooled = (weights.unsqueeze(-1) * f).sum(dim=1)                    # (B, w)
        summary = self.need_summary(needs).mean(dim=1)                     # (B, w)
        return torch.cat([pooled, summary, obs["pos"]], dim=1)


EXTRACTORS = {"mlp": TokenMLP, "attn": RelationalTokens}
