"""
Pytest configuration and shared fixtures.
"""

import numpy as np
import pytest

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode


@pytest.fixture
def rng():
    """Provide a seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def baseline_env():
    """Create a baseline environment for testing."""
    return ForagingGridEnv(
        grid_size=20,
        max_steps=1000,
        initial_energy=100.0,
        mode=EnvironmentMode.BASELINE,
        seed=42,
    )


@pytest.fixture
def small_env():
    """Create a small environment for quick tests."""
    return ForagingGridEnv(
        grid_size=10,
        max_steps=100,
        initial_energy=100.0,
        mode=EnvironmentMode.BASELINE,
        seed=42,
    )


@pytest.fixture
def path_blockage_env():
    """Create a path blockage environment for testing."""
    return ForagingGridEnv(
        grid_size=20,
        max_steps=500,
        mode=EnvironmentMode.PATH_BLOCKAGE,
        seed=42,
    )


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_sb3: requires stable-baselines3"
    )
    config.addinivalue_line(
        "markers", "requires_pymdp: requires pymdp"
    )
