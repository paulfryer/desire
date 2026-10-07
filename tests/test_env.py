import numpy as np
import pytest

from src.env import (
    BITE,
    CLAMP_STATE,
    OOB,
    SETPOINT,
    A,
    B,
    C,
    EnvConfig,
    NutrientGrid,
    ProbeSpec,
    _first_moves,
    drive,
    infer_increment,
    sample_probe_positions,
)


def probe_env(n=1, **cfg):
    return NutrientGrid(n, EnvConfig(**cfg), seed=0)


def single_probe(agent, items, types, h):
    return ProbeSpec(np.array([agent]), np.array([items]), np.array([types]), np.array([h], float))


def test_egocentric_view_places_items_relative_to_agent():
    env = probe_env()
    obs = env.reset_probe(single_probe((4, 4), [(4, 6), (1, 4)], [A, C], [0.5, 0.5, 0.7]))
    centre = env.cfg.grid - 1
    grid = obs["grid"][0]
    assert grid[A, centre, centre + 2] == 1.0
    assert grid[C, centre - 3, centre] == 1.0
    assert grid[:3].sum() == 2.0


def test_out_of_bounds_channel_marks_outside_grid():
    env = probe_env()
    obs = env.reset_probe(single_probe((0, 0), [(0, 3), (3, 0)], [A, B], [0.5, 0.5, 0.7]))
    centre = env.cfg.grid - 1
    grid = obs["grid"][0]
    assert grid[OOB, centre - 1, centre] == 1.0     # above row 0
    assert grid[OOB, centre, centre - 1] == 1.0     # left of column 0
    assert grid[OOB, centre, centre] == 0.0
    assert grid[OOB, centre + 8, centre + 8] == 0.0  # cell (8, 8) is inside


def test_eating_raises_nutrient_and_reduces_drive_when_hungry():
    env = probe_env()
    env.reset_probe(single_probe((4, 4), [(4, 5), (0, 0)], [B, A], [0.5, 0.1, SETPOINT]))
    _, reward, done, eaten = env.step(np.array([3]))  # right onto B
    assert eaten[0] == B and done[0]
    assert env.h[0, B] == pytest.approx(0.1 + BITE - 0.004)
    assert reward[0] > 0


def test_eating_when_sated_overshoots_and_is_punished():
    env = probe_env()
    env.reset_probe(single_probe((4, 4), [(4, 5), (0, 0)], [A, B], [SETPOINT, 0.5, SETPOINT]))
    _, reward, _, eaten = env.step(np.array([3]))
    assert eaten[0] == A
    assert reward[0] < 0


def test_c_never_depletes_and_relaxes_toward_setpoint():
    env = NutrientGrid(4, EnvConfig(), seed=1)
    env.reset()
    env.h[:, C] = 1.0
    for _ in range(100):
        env.step(np.full(4, 4))
    assert np.all(env.h[:, C] >= SETPOINT - 1e-9)
    assert np.all(env.h[:, C] < SETPOINT + 0.05)


def test_eating_c_at_setpoint_is_observable_and_aversive():
    # The effect of C must be recoverable (not clipped at the ceiling) when C is not needed.
    env = probe_env()
    env.reset_probe(single_probe((4, 4), [(4, 5), (0, 0)], [C, A], [0.3, 0.3, SETPOINT]))
    h_before = env.h.copy()
    _, reward, _, eaten = env.step(np.array([3]))
    assert eaten[0] == C and reward[0] < 0
    np.testing.assert_allclose(infer_increment(h_before, env.h)[0], [0.0, 0.0, BITE], atol=1e-9)


def test_training_respawn_keeps_item_counts():
    env = NutrientGrid(8, EnvConfig(), seed=2)
    env.reset()
    counts = env.items.sum(axis=(2, 3))
    rng = np.random.default_rng(0)
    for _ in range(300):
        env.step(rng.integers(0, 5, size=8))
        if env.t[0] >= env.cfg.episode_steps:
            break
    np.testing.assert_array_equal(env.items.sum(axis=(2, 3)), counts)


def test_clamped_needs_reset_every_step():
    env = NutrientGrid(4, EnvConfig(need_dynamics="clamped"), seed=3)
    env.reset()
    env.step(np.zeros(4, dtype=int))
    np.testing.assert_allclose(env.h, np.tile(CLAMP_STATE, (4, 1)))


def test_interoception_off_hides_needs():
    env = NutrientGrid(2, EnvConfig(interoception=False), seed=4)
    obs = env.reset()
    assert np.all(obs["intero"] == 0)
    assert np.any(env.h != 0)


def test_episode_ends_at_episode_steps():
    env = NutrientGrid(2, EnvConfig(episode_steps=5), seed=5)
    env.reset()
    for _ in range(4):
        *_, done, _ = env.step(np.full(2, 4))
        assert not done.any()
    *_, done, _ = env.step(np.full(2, 4))
    assert done.all()


def test_probe_positions_are_equidistant_with_disjoint_first_moves():
    rng = np.random.default_rng(6)
    agent, items = sample_probe_positions(rng, 500)
    d0 = np.abs(items[:, 0] - agent).sum(axis=1)
    d1 = np.abs(items[:, 1] - agent).sum(axis=1)
    np.testing.assert_array_equal(d0, d1)
    assert set(np.unique(d0)) <= {3, 4, 5}
    assert np.all((items >= 0) & (items < 9))
    for a, p, q in zip(agent, items[:, 0], items[:, 1], strict=True):
        assert not (_first_moves(a, p) & _first_moves(a, q))


def test_drive_is_zero_at_setpoint():
    assert drive(np.full(3, SETPOINT)) == 0.0
    assert drive(np.array([SETPOINT - 0.4, SETPOINT, SETPOINT])) == pytest.approx(0.16)
