"""
Environment implementations for the Desire research project.

Provides Gymnasium-compatible grid world environments for testing
agent behavioral signatures.
"""

from src.environments.base_env import BaseEnvironment
from src.environments.resources import Resource, ResourceType
from src.environments.terrain import TerrainType
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, get_mode_config

__all__ = [
    "BaseEnvironment",
    "Resource",
    "ResourceType",
    "TerrainType",
    "ForagingGridEnv",
    "EnvironmentMode",
    "get_mode_config",
]
