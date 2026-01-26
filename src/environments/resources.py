"""
Resource types and management for the foraging environment.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class ResourceType(Enum):
    """Types of resources available in the environment."""

    RENEWABLE = auto()      # Regenerates after being consumed
    DEPLETING = auto()      # Permanently removed when consumed
    DECAYING = auto()       # Disappears after a time limit
    PHANTOM = auto()        # Visible but provides no energy (for testing)

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Resource:
    """
    A resource in the environment that agents can consume for energy.

    Attributes:
        position: (x, y) position on the grid
        resource_type: Type of resource behavior
        energy_value: Energy gained when consumed
        regeneration_time: Steps until renewable resource regenerates
        decay_time: Steps until decaying resource disappears
        is_active: Whether the resource is currently available
        steps_since_consumed: Steps since last consumed (for regeneration)
        steps_until_decay: Steps remaining until decay (for decaying resources)
        id: Unique identifier for the resource
    """

    position: np.ndarray
    resource_type: ResourceType = ResourceType.RENEWABLE
    energy_value: float = 20.0
    regeneration_time: int = 50
    decay_time: int = 100
    is_active: bool = True
    steps_since_consumed: int = 0
    steps_until_decay: int | None = None
    id: int = field(default_factory=lambda: Resource._next_id())

    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> int:
        """Generate a unique resource ID."""
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter (useful for testing)."""
        cls._id_counter = 0

    def __post_init__(self):
        """Initialize decay timer for decaying resources."""
        if self.resource_type == ResourceType.DECAYING and self.steps_until_decay is None:
            self.steps_until_decay = self.decay_time

    def consume(self) -> float:
        """
        Consume the resource and return energy gained.

        Returns:
            Energy value if resource was active, 0 otherwise
        """
        if not self.is_active:
            return 0.0

        energy_gained = self.energy_value

        if self.resource_type == ResourceType.PHANTOM:
            # Phantom resources look real but give no energy
            energy_gained = 0.0

        if self.resource_type == ResourceType.DEPLETING:
            # Permanently removed
            self.is_active = False
        elif self.resource_type in (ResourceType.RENEWABLE, ResourceType.PHANTOM):
            # Temporarily inactive, will regenerate
            self.is_active = False
            self.steps_since_consumed = 0
        elif self.resource_type == ResourceType.DECAYING:
            # Consumed and removed
            self.is_active = False

        return energy_gained

    def update(self) -> None:
        """Update resource state for one timestep."""
        if self.resource_type == ResourceType.RENEWABLE and not self.is_active:
            # Check for regeneration
            self.steps_since_consumed += 1
            if self.steps_since_consumed >= self.regeneration_time:
                self.is_active = True
                self.steps_since_consumed = 0

        elif self.resource_type == ResourceType.DECAYING and self.is_active:
            # Check for decay
            if self.steps_until_decay is not None:
                self.steps_until_decay -= 1
                if self.steps_until_decay <= 0:
                    self.is_active = False

    def reset(self) -> None:
        """Reset the resource to its initial state."""
        self.is_active = True
        self.steps_since_consumed = 0
        if self.resource_type == ResourceType.DECAYING:
            self.steps_until_decay = self.decay_time

    @property
    def x(self) -> int:
        """X coordinate of the resource."""
        return int(self.position[0])

    @property
    def y(self) -> int:
        """Y coordinate of the resource."""
        return int(self.position[1])

    def distance_to(self, position: np.ndarray) -> float:
        """Calculate Manhattan distance to a position."""
        return float(np.abs(self.position - position).sum())

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resource):
            return False
        return self.id == other.id


def create_resource_cluster(
    center: np.ndarray,
    resource_type: ResourceType,
    count: int = 5,
    spread: int = 3,
    rng: np.random.Generator | None = None,
    grid_size: int = 20,
    energy_value: float = 20.0,
) -> list[Resource]:
    """
    Create a cluster of resources around a center point.

    Args:
        center: Center position for the cluster
        resource_type: Type of resources to create
        count: Number of resources in the cluster
        spread: Maximum distance from center
        rng: Random number generator
        grid_size: Size of the grid (for bounds checking)
        energy_value: Energy value for each resource

    Returns:
        List of Resource objects
    """
    if rng is None:
        rng = np.random.default_rng()

    resources = []
    attempts = 0
    max_attempts = count * 10

    while len(resources) < count and attempts < max_attempts:
        # Generate random offset
        offset = rng.integers(-spread, spread + 1, size=2)
        position = center + offset

        # Bounds check
        if (position >= 0).all() and (position < grid_size).all():
            # Check for duplicates
            pos_tuple = tuple(position)
            if not any(tuple(r.position) == pos_tuple for r in resources):
                resources.append(Resource(
                    position=position.copy(),
                    resource_type=resource_type,
                    energy_value=energy_value,
                ))

        attempts += 1

    return resources
