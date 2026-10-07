"""
Model-free agents: stable-baselines3 PPO and recurrent PPO trained on NutrientGrid.
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from src.agents.base import Agent
from src.env import N_ACTIONS, N_TYPES, EnvConfig, NutrientGrid


def observation_space(config: EnvConfig) -> spaces.Dict:
    v = config.view
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(N_TYPES + 1, v, v), dtype=np.float32),
        "intero": spaces.Box(0.0, 1.0, shape=(N_TYPES,), dtype=np.float32),
    })


class NutrientVecEnv(VecEnv):
    """stable-baselines3 VecEnv over the batched NutrientGrid (auto-resets finished episodes)."""

    def __init__(self, n_envs: int, config: EnvConfig, seed: int | None = None):
        self.grid_env = NutrientGrid(n_envs, config, seed)
        super().__init__(n_envs, observation_space(config), spaces.Discrete(N_ACTIONS))
        self._actions = np.zeros(n_envs, dtype=np.int64)
        self._returns = np.zeros(n_envs)
        self._lengths = np.zeros(n_envs, dtype=np.int64)
        self._eaten = np.zeros((n_envs, N_TYPES), dtype=np.int64)

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
                # Episodes only end by time limit, so value bootstrapping is correct.
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


class GridFeatures(BaseFeaturesExtractor):
    """Small CNN over the egocentric item grid, concatenated with the interoceptive vector."""

    def __init__(self, observation_space: spaces.Dict, cnn_dim: int = 256):
        channels, v, _ = observation_space["grid"].shape
        n_intero = observation_space["intero"].shape[0]
        super().__init__(observation_space, features_dim=cnn_dim + n_intero)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, channels, v, v)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, cnn_dim), nn.ReLU())

    def forward(self, observations):
        return torch.cat([self.linear(self.cnn(observations["grid"])), observations["intero"]], dim=1)


class SB3Agent(Agent):
    """Wraps a trained stable-baselines3 model for batched acting in assays."""

    def __init__(self, model, name: str, interoception: bool, recurrent: bool = False,
                 deterministic: bool = False):
        self.model = model
        self.name = name
        self.interoception = interoception
        self.recurrent = recurrent
        self.deterministic = deterministic
        self._state = None
        self._starts = None

    def reset(self, n):
        self._state = None
        self._starts = np.ones(n, dtype=bool)

    def act(self, obs):
        if self.recurrent:
            actions, self._state = self.model.predict(
                obs, state=self._state, episode_start=self._starts, deterministic=self.deterministic)
            self._starts[:] = False
            return actions
        actions, _ = self.model.predict(obs, deterministic=self.deterministic)
        return actions
