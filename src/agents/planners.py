"""
Option-based planners (docs/RESEARCH_DESIGN_v2.md §5).

All planners share the same options (go to a visible item by shortest path and eat it, or stay),
the same planning horizon, innate knowledge of the body's own drift (`body_drift`), and the
same epsilon exploration over options during training. They differ only in how they value the
predicted outcome of an option:

- HomeostaticPlanner (`homeo`): learns the mean effect of each item type on h, scores options
  by the innate drive D summed over the horizon. With the true effects it is the `oracle`.
- RewardModelPlanner (`mb_reward`): learns the same effects plus a scalar reward model
  r(h, event) from experienced rewards, and scores options by predicted summed reward.
- ActiveInferencePlanner (`aif`): learns Dirichlet distributions over each type's effect on
  each nutrient, and scores options by expected free energy: risk (KL from predicted
  interoceptive observations to preferred ones) minus novelty (expected information gain
  about the Dirichlet parameters). Novelty is only used while learning (Amendment 1).
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.agents.base import Agent, Option, parse_options
from src.env import BITE, N_TYPES, SETPOINT, body_drift, infer_increment

HORIZON = 20
EVENTS = N_TYPES + 1  # no-eat, A, B, C


class OptionPlanner(Agent):
    """Shared machinery: option enumeration, outcome simulation, exploration and learning hooks."""

    def __init__(self, epsilon: float = 0.1, seed: int | None = None, horizon: int = HORIZON):
        self.epsilon = epsilon
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.learning = True
        self.n_consumed = np.zeros(N_TYPES, dtype=np.int64)
        self.effect_sum = np.zeros((N_TYPES, N_TYPES))
        self.effect_n = np.zeros((N_TYPES, N_TYPES))

    # -------------------------------------------------------------- acting

    def freeze(self) -> None:
        """Stop learning and exploring (used for assays)."""
        self.learning = False

    def act(self, obs):
        actions = np.empty(len(obs["intero"]), dtype=np.int64)
        for b in range(len(actions)):
            options = parse_options(obs["grid"][b])
            actions[b] = self.choose(options, obs["intero"][b].astype(np.float64)).first_action
        return actions

    def choose(self, options: list[Option], h: np.ndarray) -> Option:
        if self.learning and self.rng.random() < self.epsilon:
            return options[self.rng.integers(len(options))]
        scores = self.score(options, h)       # lower is better
        best = np.flatnonzero(scores <= scores.min() + 1e-9)
        return options[self.rng.choice(best)]

    def score(self, options: list[Option], h: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -------------------------------------------------------------- prediction

    def mean_effects(self) -> np.ndarray:
        """Learned mean effect of eating each type on each nutrient (0 where never observed)."""
        with np.errstate(invalid="ignore"):
            mean = self.effect_sum / self.effect_n
        return np.nan_to_num(mean)

    def simulate(self, options: list[Option], h: np.ndarray, effects: np.ndarray):
        """Deterministic rollout of h under each option.

        Returns h_before (O, H, 3), events (O, H) in 0..3 (0 = no eating), h_after (O, H, 3)."""
        n_opt = len(options)
        states = np.tile(h, (n_opt, 1))
        before = np.zeros((n_opt, self.horizon, N_TYPES))
        after = np.zeros_like(before)
        events = np.zeros((n_opt, self.horizon), dtype=np.int64)
        eat_step = np.array([o.distance if o.item_type is not None else -1 for o in options])
        eat_type = np.array([o.item_type if o.item_type is not None else -1 for o in options])
        for tau in range(1, self.horizon + 1):
            before[:, tau - 1] = states
            eating = eat_step == tau
            if eating.any():
                states[eating] = np.minimum(1.0, states[eating] + effects[eat_type[eating]])
                events[eating, tau - 1] = eat_type[eating] + 1
            states = body_drift(states)
            after[:, tau - 1] = states
        return before, events, after

    # -------------------------------------------------------------- learning

    def learn(self, h_before: np.ndarray, eaten: np.ndarray, h_after: np.ndarray,
              reward: np.ndarray) -> None:
        """Update from one batched environment step (no-op when frozen)."""
        if not self.learning:
            return
        for b in np.flatnonzero(eaten >= 0):
            k = eaten[b]
            self.n_consumed[k] += 1
            delta = infer_increment(h_before[b].astype(np.float64), h_after[b].astype(np.float64))
            seen = ~np.isnan(delta)
            self.effect_sum[k, seen] += delta[seen]
            self.effect_n[k, seen] += 1
            self._learn_effect(k, delta)
        self._learn_reward(h_before, eaten, reward)

    def _learn_effect(self, k: int, delta: np.ndarray) -> None:
        pass

    def _learn_reward(self, h_before, eaten, reward) -> None:
        pass

    def end_of_round(self) -> None:
        """Called after each batch of training episodes."""


class HomeostaticPlanner(OptionPlanner):
    """Values options by the innate drive D(h) summed over the horizon (lower is better)."""

    name = "homeo"

    def __init__(self, *args, true_effects: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_effects = true_effects
        if true_effects:
            self.name = "oracle"
            self.learning = False

    def effects(self) -> np.ndarray:
        if self.true_effects:
            return BITE * np.eye(N_TYPES)
        return self.mean_effects()

    def score(self, options, h):
        _, _, after = self.simulate(options, h, self.effects())
        return ((SETPOINT - after) ** 2).sum(axis=(1, 2))


class RewardModelPlanner(OptionPlanner):
    """Learns a scalar reward model r(h, event) and values options by predicted summed reward."""

    name = "mb_reward"

    def __init__(self, *args, hidden=(64, 64), max_buffer: int = 200_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden = hidden
        self.max_buffer = max_buffer
        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self.model: MLPRegressor | None = None

    @staticmethod
    def features(h: np.ndarray, events: np.ndarray) -> np.ndarray:
        onehot = np.eye(EVENTS)[events]
        return np.concatenate([h, onehot], axis=-1)

    def _learn_reward(self, h_before, eaten, reward):
        self._x.append(self.features(h_before.astype(np.float64), eaten + 1))
        self._y.append(reward.astype(np.float64))

    def end_of_round(self):
        if not self.learning:
            return
        x = np.concatenate(self._x)[-self.max_buffer:]
        y = np.concatenate(self._y)[-self.max_buffer:]
        # Eating events are rare; balance them against no-eat steps so the model learns them.
        eat = np.flatnonzero(x[:, N_TYPES] == 0)
        rest = np.flatnonzero(x[:, N_TYPES] == 1)
        if eat.size == 0:
            return
        keep = np.concatenate([eat, self.rng.choice(rest, size=min(rest.size, 2 * eat.size),
                                                    replace=False)])
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=300,
                                  early_stopping=True, random_state=int(self.rng.integers(2**31)))
        self.model.fit(x[keep], y[keep])

    def score(self, options, h):
        if self.model is None:
            return self.rng.random(len(options))
        before, events, _ = self.simulate(options, h, self.mean_effects())
        x = self.features(before, events).reshape(-1, N_TYPES + EVENTS)
        predicted = self.model.predict(x).reshape(len(options), self.horizon)
        return -predicted.sum(axis=1)


class ActiveInferencePlanner(OptionPlanner):
    """Expected free energy over options with Dirichlet-learned outcome distributions."""

    name = "aif"
    BIN = 0.05
    N_LEVELS = 21          # interoceptive observation levels 0, 0.05, ..., 1.0
    N_DELTAS = 13          # outcome increments -0.30 ... +0.30

    def __init__(self, *args, novelty: bool = True, preference_precision: float = 50.0,
                 policy_precision: float = 1.0, prior_concentration: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.novelty = novelty
        if not novelty:
            self.name = "aif_no_novelty"
        self.policy_precision = policy_precision
        levels = np.arange(self.N_LEVELS) * self.BIN
        log_pref = -preference_precision * (levels - SETPOINT) ** 2
        self.log_pref = log_pref - np.log(np.exp(log_pref).sum())   # log p~(o), same per nutrient
        self.deltas = (np.arange(self.N_DELTAS) - self.N_DELTAS // 2) * self.BIN
        self.counts = np.full((N_TYPES, N_TYPES, self.N_DELTAS), prior_concentration)

    def _learn_effect(self, k, delta):
        for i in np.flatnonzero(~np.isnan(delta)):
            m = int(np.clip(np.rint(delta[i] / self.BIN) + self.N_DELTAS // 2, 0, self.N_DELTAS - 1))
            self.counts[k, i, m] += 1

    def choose(self, options, h):
        if self.learning and self.rng.random() < self.epsilon:
            return options[self.rng.integers(len(options))]
        g = self.score(options, h)
        logits = -self.policy_precision * (g - g.min())
        p = np.exp(logits)
        return options[self.rng.choice(len(options), p=p / p.sum())]

    def expected_free_energy(self, options, h) -> tuple[np.ndarray, np.ndarray]:
        """Returns (risk, novelty) per option, each summed over the horizon."""
        n_opt, horizon, nd = len(options), self.horizon, self.N_DELTAS
        predictive = self.counts / self.counts.sum(axis=-1, keepdims=True)   # (type, nutrient, delta)
        # Point masses per (option, nutrient, delta); before eating all mass sits on delta = 0.
        points = np.tile(h[None, :, None], (n_opt, 1, nd)).astype(np.float64)
        weights = np.zeros((n_opt, N_TYPES, nd))
        weights[:, :, nd // 2] = 1.0
        novelty = np.zeros(n_opt)
        risk = np.zeros(n_opt)
        eat_step = np.array([o.distance if o.item_type is not None else -1 for o in options])
        for o, opt in enumerate(options):
            if opt.item_type is not None:
                a = self.counts[opt.item_type]
                w = 0.5 * (1.0 / a - 1.0 / a.sum(axis=-1, keepdims=True))
                novelty[o] = (predictive[opt.item_type] * w).sum()
        level_onehot = np.eye(self.N_LEVELS)
        for tau in range(1, horizon + 1):
            for o in np.flatnonzero(eat_step == tau):
                k = options[o].item_type
                points[o] = np.minimum(1.0, points[o][:, nd // 2:nd // 2 + 1] + self.deltas[None, :])
                points[o] = np.maximum(points[o], 0.0)
                weights[o] = predictive[k]
            points = body_drift(points.transpose(0, 2, 1)).transpose(0, 2, 1)
            bins = np.clip(np.rint(points / self.BIN).astype(np.int64), 0, self.N_LEVELS - 1)
            q = np.einsum("oid,oidl->oil", weights, level_onehot[bins])          # (O, 3, levels)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.where(q > 0, q * (np.log(q) - self.log_pref), 0.0)
            risk += kl.sum(axis=(1, 2))
        return risk, novelty

    def score(self, options, h):
        risk, novelty = self.expected_free_energy(options, h)
        if self.novelty and self.learning:
            return risk - novelty
        return risk


def make_oracle(seed: int | None = None) -> HomeostaticPlanner:
    return HomeostaticPlanner(epsilon=0.0, seed=seed, true_effects=True)
