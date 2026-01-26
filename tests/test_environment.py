"""
Unit tests for environment components.
"""

import numpy as np

from src.environments.foraging_grid import Action, ForagingGridEnv
from src.environments.modes import EnvironmentMode, get_mode_config
from src.environments.resources import Resource, ResourceType, create_resource_cluster
from src.environments.terrain import TerrainGrid, TerrainType


class TestResource:
    """Tests for Resource class."""

    def test_resource_creation(self):
        """Test basic resource creation."""
        pos = np.array([5, 5])
        resource = Resource(position=pos, resource_type=ResourceType.RENEWABLE)

        assert resource.x == 5
        assert resource.y == 5
        assert resource.is_active
        assert resource.energy_value == 20.0

    def test_renewable_consumption(self):
        """Test renewable resource consumption and regeneration."""
        resource = Resource(
            position=np.array([0, 0]),
            resource_type=ResourceType.RENEWABLE,
            regeneration_time=5,
        )

        # Consume
        energy = resource.consume()
        assert energy == 20.0
        assert not resource.is_active

        # Wait for regeneration
        for _ in range(4):
            resource.update()
            assert not resource.is_active

        resource.update()
        assert resource.is_active

    def test_depleting_consumption(self):
        """Test depleting resource is permanently removed."""
        resource = Resource(
            position=np.array([0, 0]),
            resource_type=ResourceType.DEPLETING,
        )

        energy = resource.consume()
        assert energy == 20.0
        assert not resource.is_active

        # Should not regenerate
        for _ in range(100):
            resource.update()
        assert not resource.is_active

    def test_phantom_resource(self):
        """Test phantom resource gives no energy."""
        resource = Resource(
            position=np.array([0, 0]),
            resource_type=ResourceType.PHANTOM,
        )

        energy = resource.consume()
        assert energy == 0.0

    def test_decaying_resource(self):
        """Test decaying resource disappears over time."""
        resource = Resource(
            position=np.array([0, 0]),
            resource_type=ResourceType.DECAYING,
            decay_time=10,
        )

        assert resource.is_active
        for _ in range(9):
            resource.update()
            assert resource.is_active

        resource.update()
        assert not resource.is_active

    def test_resource_cluster_creation(self):
        """Test cluster creation."""
        rng = np.random.default_rng(42)
        center = np.array([10, 10])

        cluster = create_resource_cluster(
            center=center,
            resource_type=ResourceType.RENEWABLE,
            count=5,
            spread=3,
            rng=rng,
            grid_size=20,
        )

        assert len(cluster) == 5
        for resource in cluster:
            assert abs(resource.x - 10) <= 3
            assert abs(resource.y - 10) <= 3


class TestTerrain:
    """Tests for TerrainGrid class."""

    def test_terrain_initialization(self):
        """Test terrain grid initialization."""
        terrain = TerrainGrid(size=10)

        assert terrain.size == 10
        for x in range(10):
            for y in range(10):
                assert terrain[x, y] == TerrainType.OPEN
                assert terrain.is_passable(x, y)

    def test_obstacle_placement(self):
        """Test obstacle placement."""
        terrain = TerrainGrid(size=10)
        terrain[5, 5] = TerrainType.OBSTACLE

        assert not terrain.is_passable(5, 5)
        assert terrain.is_passable(5, 4)

    def test_wall_creation(self):
        """Test wall creation."""
        terrain = TerrainGrid(size=10)
        terrain.add_wall((2, 2), (2, 8))

        for y in range(2, 9):
            assert not terrain.is_passable(2, y)

    def test_dynamic_obstacle(self):
        """Test dynamic obstacle toggling."""
        terrain = TerrainGrid(size=10)
        terrain[5, 5] = TerrainType.DYNAMIC_OBSTACLE

        # Initially blocked
        assert not terrain.is_passable(5, 5)

        # Toggle
        terrain.toggle_dynamic_obstacle(5, 5)
        assert terrain.is_passable(5, 5)

        terrain.toggle_dynamic_obstacle(5, 5)
        assert not terrain.is_passable(5, 5)

    def test_slow_terrain_cost(self):
        """Test slow terrain movement cost."""
        terrain = TerrainGrid(size=10, slow_cost=2.0)
        terrain[3, 3] = TerrainType.SLOW

        assert terrain.get_movement_cost(3, 3) == 2.0
        assert terrain.get_movement_cost(4, 4) == 0.0

    def test_bounds_checking(self):
        """Test bounds checking."""
        terrain = TerrainGrid(size=10)

        assert terrain.in_bounds(0, 0)
        assert terrain.in_bounds(9, 9)
        assert not terrain.in_bounds(-1, 0)
        assert not terrain.in_bounds(10, 0)
        assert not terrain.is_passable(-1, 0)


class TestForagingGridEnv:
    """Tests for ForagingGridEnv."""

    def test_env_creation(self):
        """Test environment creation."""
        env = ForagingGridEnv(grid_size=20, max_steps=100, seed=42)

        assert env.grid_size == 20
        assert env.max_steps == 100
        assert env.action_space.n == 6

    def test_reset(self):
        """Test environment reset."""
        env = ForagingGridEnv(grid_size=20, seed=42)
        obs, info = env.reset()

        assert "local_view" in obs
        assert "energy" in obs
        assert "position" in obs
        assert "step" in obs

        assert obs["energy"][0] == 1.0  # Full energy at start
        assert env.current_step == 0

    def test_movement_actions(self):
        """Test movement actions work correctly."""
        env = ForagingGridEnv(grid_size=20, seed=42)
        env.reset()

        initial_pos = env.agent_position.copy()

        # Move down
        obs, reward, terminated, truncated, info = env.step(Action.DOWN)
        expected_pos = initial_pos + np.array([1, 0])

        # Check if movement occurred (might be blocked)
        if env.terrain.is_passable(expected_pos[0], expected_pos[1]):
            assert np.array_equal(env.agent_position, expected_pos)

    def test_energy_decay(self):
        """Test energy decays each step."""
        env = ForagingGridEnv(
            grid_size=20,
            initial_energy=100.0,
            energy_decay_rate=1.0,
            seed=42,
        )
        env.reset()

        initial_energy = env.agent_energy
        env.step(Action.STAY)

        assert env.agent_energy == initial_energy - 1.0

    def test_consume_action(self):
        """Test consuming resource gives energy."""
        env = ForagingGridEnv(grid_size=20, seed=42)
        obs, info = env.reset()

        # Place agent on a resource
        if env.resources:
            env.agent_position = env.resources[0].position.copy()
            env.agent_energy = 50.0  # Low energy

            obs, reward, terminated, truncated, info = env.step(Action.CONSUME)

            assert env.agent_energy > 50.0 or not env.resources[0].is_active

    def test_death_on_zero_energy(self):
        """Test episode terminates on zero energy."""
        env = ForagingGridEnv(
            grid_size=20,
            initial_energy=5.0,
            energy_decay_rate=2.0,
            seed=42,
        )
        env.reset()

        # Drain energy
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(Action.STAY)
            if terminated:
                break

        assert terminated
        assert env.agent_energy <= 0

    def test_truncation_on_max_steps(self):
        """Test episode truncates at max steps."""
        env = ForagingGridEnv(
            grid_size=20,
            max_steps=10,
            energy_decay_rate=0.1,  # Slow decay to survive
            seed=42,
        )
        env.reset()

        for _i in range(15):
            obs, reward, terminated, truncated, info = env.step(Action.STAY)
            if truncated:
                break

        assert truncated
        assert env.current_step >= 10

    def test_observation_shape(self):
        """Test observation has correct shape."""
        env = ForagingGridEnv(
            grid_size=20,
            local_view_size=5,
            seed=42,
        )
        obs, info = env.reset()

        view_dim = 2 * 5 + 1  # 11
        assert obs["local_view"].shape == (view_dim, view_dim, 3)
        assert obs["energy"].shape == (1,)
        assert obs["position"].shape == (2,)
        assert obs["step"].shape == (1,)

    def test_render_modes(self):
        """Test different render modes."""
        env = ForagingGridEnv(grid_size=10, render_mode="ansi", seed=42)
        env.reset()

        ansi_output = env.render()
        assert isinstance(ansi_output, str)
        assert "A" in ansi_output  # Agent marker

        env.close()


class TestEnvironmentModes:
    """Tests for environment modes."""

    def test_baseline_mode(self):
        """Test baseline mode configuration."""
        config = get_mode_config(EnvironmentMode.BASELINE, grid_size=20)

        assert config.mode == EnvironmentMode.BASELINE
        assert config.num_resources == 5
        assert len(config.resource_types) == 5

    def test_path_blockage_mode(self):
        """Test path blockage mode."""
        env = ForagingGridEnv(
            grid_size=20,
            mode=EnvironmentMode.PATH_BLOCKAGE,
            energy_decay_rate=0.1,  # Slow decay to survive long enough
            seed=42,
        )
        env.reset()

        # Initially path should be open
        assert not env.path_blocked

        # Step past blockage point
        for _ in range(env.mode_config.blockage_step + 5):
            obs, reward, terminated, truncated, info = env.step(Action.STAY)
            if terminated or truncated:
                break

        assert env.path_blocked

    def test_extinction_mode(self):
        """Test extinction mode has depleting resources."""
        config = get_mode_config(EnvironmentMode.EXTINCTION, grid_size=20)

        assert all(rt == "depleting" for rt in config.resource_types)

    def test_phantom_mode(self):
        """Test phantom mode has phantom resources."""
        config = get_mode_config(EnvironmentMode.PHANTOM, grid_size=20)

        assert "phantom" in config.resource_types


class TestGymCompatibility:
    """Test Gymnasium API compatibility."""

    def test_gym_api_compliance(self):
        """Test environment follows Gymnasium API."""
        env = ForagingGridEnv(grid_size=20, seed=42)

        # Check required attributes
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')

    def test_reset_returns_tuple(self):
        """Test reset returns (observation, info) tuple."""
        env = ForagingGridEnv(grid_size=20, seed=42)
        result = env.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(info, dict)

    def test_step_returns_tuple(self):
        """Test step returns 5-tuple."""
        env = ForagingGridEnv(grid_size=20, seed=42)
        env.reset()
        result = env.step(0)

        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_seeding(self):
        """Test environment seeding produces reproducible results."""
        env1 = ForagingGridEnv(grid_size=20, seed=42)
        env2 = ForagingGridEnv(grid_size=20, seed=42)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1["local_view"], obs2["local_view"])
        np.testing.assert_array_equal(obs1["position"], obs2["position"])
