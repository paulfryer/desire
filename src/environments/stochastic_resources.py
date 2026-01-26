"""
Stochastic resource dynamics for foraging environments.

Provides environments where resources have unpredictable behavior:
- Random spawning/despawning
- Variable energy values
- Moving resources
- Resource clusters with correlations
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Wrapper

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.resources import Resource, ResourceType


class StochasticResourceWrapper(Wrapper):
    """
    Wrapper that adds stochastic resource dynamics.

    Features:
    - Resources can spawn and despawn randomly
    - Energy values can vary
    - Resources can move
    """

    def __init__(
        self,
        env: ForagingGridEnv,
        spawn_rate: float = 0.01,
        despawn_rate: float = 0.01,
        move_rate: float = 0.0,
        energy_variance: float = 0.0,
        max_resources: int = 20,
        min_resources: int = 1,
    ):
        """
        Initialize stochastic resource wrapper.

        Args:
            env: Base foraging environment
            spawn_rate: Probability of spawning new resource per step
            despawn_rate: Probability of despawning each resource per step
            move_rate: Probability of each resource moving per step
            energy_variance: Variance in energy values (as fraction of mean)
            max_resources: Maximum number of resources
            min_resources: Minimum number of resources
        """
        super().__init__(env)

        self.spawn_rate = spawn_rate
        self.despawn_rate = despawn_rate
        self.move_rate = move_rate
        self.energy_variance = energy_variance
        self.max_resources = max_resources
        self.min_resources = min_resources

        self._rng = np.random.default_rng()

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        if "seed" in kwargs:
            self._rng = np.random.default_rng(kwargs["seed"])

        obs, info = self.env.reset(**kwargs)
        info["resource_dynamics"] = "stochastic"
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with stochastic resource updates."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update resources stochastically
        self._update_resources()

        info["n_active_resources"] = sum(1 for r in self.env.resources if r.is_active)

        return obs, reward, terminated, truncated, info

    def _update_resources(self) -> None:
        """Apply stochastic updates to resources."""
        # Despawn resources
        active_count = sum(1 for r in self.env.resources if r.is_active)

        for resource in self.env.resources:
            if resource.is_active and active_count > self.min_resources:
                if self._rng.random() < self.despawn_rate:
                    resource.is_active = False
                    active_count -= 1

        # Spawn new resources
        if active_count < self.max_resources:
            if self._rng.random() < self.spawn_rate:
                self._spawn_resource()

        # Move resources
        if self.move_rate > 0:
            self._move_resources()

        # Vary energy values
        if self.energy_variance > 0:
            self._vary_energy()

    def _spawn_resource(self) -> None:
        """Spawn a new resource at random valid location."""
        valid_positions = self.env.terrain.get_passable_positions()

        # Exclude positions with resources or agent
        occupied = {tuple(r.position) for r in self.env.resources}
        occupied.add(tuple(self.env.agent_position))
        valid_positions = [p for p in valid_positions if p not in occupied]

        if valid_positions:
            pos = valid_positions[self._rng.integers(len(valid_positions))]

            # Either reactivate an inactive resource or create new
            for resource in self.env.resources:
                if not resource.is_active:
                    resource.position = np.array(pos)
                    resource.is_active = True
                    resource.cooldown = 0
                    return

            # Create new if no inactive
            self.env.resources.append(Resource(
                position=np.array(pos),
                resource_type=ResourceType.RENEWABLE,
                energy_value=20.0,
            ))

    def _move_resources(self) -> None:
        """Move resources randomly."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for resource in self.env.resources:
            if resource.is_active and self._rng.random() < self.move_rate:
                dx, dy = directions[self._rng.integers(4)]
                new_pos = resource.position + np.array([dx, dy])

                # Check if valid
                if (self.env.terrain.is_passable(new_pos[0], new_pos[1]) and
                        tuple(new_pos) != tuple(self.env.agent_position)):
                    resource.position = new_pos

    def _vary_energy(self) -> None:
        """Vary resource energy values."""
        for resource in self.env.resources:
            if resource.is_active:
                base_energy = 20.0
                variation = self._rng.normal(0, base_energy * self.energy_variance)
                resource.energy_value = max(5.0, base_energy + variation)


class ResourceClusterWrapper(Wrapper):
    """
    Wrapper that creates correlated resource clusters.

    Resources appear in clusters where spawning/despawning is correlated.
    """

    def __init__(
        self,
        env: ForagingGridEnv,
        n_clusters: int = 3,
        cluster_radius: int = 3,
        cluster_spawn_rate: float = 0.05,
        cluster_despawn_rate: float = 0.03,
    ):
        """
        Initialize resource cluster wrapper.

        Args:
            env: Base environment
            n_clusters: Number of resource clusters
            cluster_radius: Radius of each cluster
            cluster_spawn_rate: Probability of cluster-wide spawn
            cluster_despawn_rate: Probability of cluster-wide despawn
        """
        super().__init__(env)

        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.cluster_spawn_rate = cluster_spawn_rate
        self.cluster_despawn_rate = cluster_despawn_rate

        self._rng = np.random.default_rng()
        self.cluster_centers: List[Tuple[int, int]] = []

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with cluster initialization."""
        if "seed" in kwargs:
            self._rng = np.random.default_rng(kwargs["seed"])

        obs, info = self.env.reset(**kwargs)

        # Initialize cluster centers
        self._initialize_clusters()

        info["n_clusters"] = self.n_clusters
        return obs, info

    def _initialize_clusters(self) -> None:
        """Initialize cluster center positions."""
        self.cluster_centers = []
        grid = self.env.grid_size

        for _ in range(self.n_clusters):
            # Random center avoiding edges
            x = self._rng.integers(self.cluster_radius, grid - self.cluster_radius)
            y = self._rng.integers(self.cluster_radius, grid - self.cluster_radius)
            self.cluster_centers.append((x, y))

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with cluster dynamics."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update clusters
        self._update_clusters()

        info["cluster_centers"] = self.cluster_centers

        return obs, reward, terminated, truncated, info

    def _update_clusters(self) -> None:
        """Update resources based on cluster dynamics."""
        for center in self.cluster_centers:
            # Decide cluster-wide action
            if self._rng.random() < self.cluster_spawn_rate:
                self._spawn_in_cluster(center)
            elif self._rng.random() < self.cluster_despawn_rate:
                self._despawn_in_cluster(center)

    def _spawn_in_cluster(self, center: Tuple[int, int]) -> None:
        """Spawn resources in cluster area."""
        cx, cy = center

        for _ in range(self._rng.integers(1, 3)):
            dx = self._rng.integers(-self.cluster_radius, self.cluster_radius + 1)
            dy = self._rng.integers(-self.cluster_radius, self.cluster_radius + 1)
            x, y = cx + dx, cy + dy

            if self.env.terrain.is_passable(x, y):
                # Check if already has resource
                if not any(tuple(r.position) == (x, y) for r in self.env.resources):
                    self.env.resources.append(Resource(
                        position=np.array([x, y]),
                        resource_type=ResourceType.RENEWABLE,
                        energy_value=20.0,
                    ))

    def _despawn_in_cluster(self, center: Tuple[int, int]) -> None:
        """Despawn resources in cluster area."""
        cx, cy = center

        for resource in self.env.resources:
            if resource.is_active:
                dist = np.sqrt((resource.x - cx) ** 2 + (resource.y - cy) ** 2)
                if dist <= self.cluster_radius:
                    if self._rng.random() < 0.5:
                        resource.is_active = False


class SeasonalResourceWrapper(Wrapper):
    """
    Wrapper that adds seasonal resource availability patterns.

    Resource density varies cyclically over time.
    """

    def __init__(
        self,
        env: ForagingGridEnv,
        season_length: int = 200,
        abundance_amplitude: float = 0.5,
        min_abundance: float = 0.3,
    ):
        """
        Initialize seasonal resource wrapper.

        Args:
            env: Base environment
            season_length: Steps per season cycle
            abundance_amplitude: Amplitude of abundance variation
            min_abundance: Minimum resource abundance fraction
        """
        super().__init__(env)

        self.season_length = season_length
        self.abundance_amplitude = abundance_amplitude
        self.min_abundance = min_abundance

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with seasonal updates."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current season phase
        phase = (self.env.current_step % self.season_length) / self.season_length
        abundance = self.min_abundance + self.abundance_amplitude * (1 + np.sin(2 * np.pi * phase)) / 2

        # Adjust resource activity based on season
        n_resources = len(self.env.resources)
        target_active = int(n_resources * abundance)

        # Count active resources
        active = [r for r in self.env.resources if r.is_active]
        inactive = [r for r in self.env.resources if not r.is_active]

        # Activate or deactivate to match target
        if len(active) < target_active and inactive:
            # Activate some
            n_activate = min(target_active - len(active), len(inactive))
            for i in range(n_activate):
                inactive[i].is_active = True
                inactive[i].cooldown = 0
        elif len(active) > target_active:
            # Deactivate some
            n_deactivate = len(active) - target_active
            for i in range(n_deactivate):
                active[i].is_active = False

        info["season_phase"] = phase
        info["abundance"] = abundance

        return obs, reward, terminated, truncated, info
