import numpy as np
import pytest

from src.agents.base import STAY, parse_options
from src.agents.controls import FixedHabitAgent, NearestAgent, NeedNoveltyAgent, RandomAgent
from src.agents.model_free import NutrientVecEnv
from src.agents.planners import ActiveInferencePlanner, HomeostaticPlanner, make_oracle
from src.assays import assay_t1, assay_t2
from src.env import BITE, SETPOINT, A, B, C, EnvConfig, NutrientGrid, ProbeSpec
from src.stats import bootstrap_ci, holm, verdict


def probe_obs(agent, items, types, h):
    env = NutrientGrid(1, EnvConfig(), seed=0)
    spec = ProbeSpec(np.array([agent]), np.array([items]), np.array([types]), np.array([h], float))
    return env, env.reset_probe(spec)


def test_parse_options_lists_items_with_distances_and_first_moves():
    _, obs = probe_obs((4, 4), [(4, 7), (1, 4)], [A, C], [0.3, 0.3, SETPOINT])
    options = {o.item_type: o for o in parse_options(obs["grid"][0])}
    assert options[None].first_action == STAY
    assert options[A].distance == 3 and options[A].first_action == 3   # right
    assert options[C].distance == 3 and options[C].first_action == 0   # up


def test_parse_options_routes_around_other_items():
    # B sits directly between the agent and A, so the path to A must detour (distance 4).
    _, obs = probe_obs((4, 4), [(4, 6), (4, 5)], [A, B], [0.3, 0.3, SETPOINT])
    options = {o.item_type: o for o in parse_options(obs["grid"][0])}
    assert options[B].distance == 1
    assert options[A].distance == 4
    assert options[A].first_action in (0, 1)


def test_oracle_and_controls_on_assays():
    oracle = make_oracle(0)
    t1, t2 = assay_t1(oracle, 30, seed=1), assay_t2(oracle, 30, seed=1)
    assert t1["DI"] == 1.0 and t2["RI"] == 1.0 and t2["RI_spec"] == 1.0
    for control in (NearestAgent(0), FixedHabitAgent(0)):
        assert abs(assay_t2(control, 30, seed=1)["RI_spec"]) < 1e-9
    assert assay_t1(FixedHabitAgent(0), 30, seed=1)["DI"] == 0.0
    novelty = assay_t2(NeedNoveltyAgent(0), 30, seed=1)
    assert novelty["RI"] == 1.0 and novelty["RI_spec"] == 0.0


def test_random_agent_rarely_chooses_within_probe():
    assert assay_t1(RandomAgent(0), 50, seed=1)["choice_rate"] < 0.6


def test_homeostatic_planner_learns_item_effects():
    agent = HomeostaticPlanner(seed=0)
    env = NutrientGrid(4, EnvConfig(), seed=0)
    for _ in range(3):
        obs = env.reset()
        done = np.zeros(4, dtype=bool)
        while not done.all():
            h_before = obs["intero"].copy()
            obs, reward, done, eaten = env.step(agent.act(obs))
            agent.learn(h_before, eaten, obs["intero"], reward)
    learned = agent.mean_effects()
    for k in range(3):
        if agent.n_consumed[k] > 0:
            assert learned[k, k] == pytest.approx(BITE, abs=1e-5)


def test_active_inference_prefers_needed_item_with_true_model():
    agent = ActiveInferencePlanner(seed=0)
    agent.counts[:] = 1e-3
    for k in range(3):
        for i in range(3):
            agent.counts[k, i, agent.N_DELTAS // 2 + (6 if i == k else 0)] = 100.0
    agent.freeze()
    _, obs = probe_obs((4, 4), [(4, 7), (1, 4)], [A, B], [SETPOINT, 0.15, SETPOINT])
    options = parse_options(obs["grid"][0])
    risk, _ = agent.expected_free_energy(options, obs["intero"][0].astype(float))
    best = options[int(np.argmin(risk))]
    assert best.item_type == B


def test_vec_env_reports_terminal_observation():
    env = NutrientVecEnv(2, EnvConfig(episode_steps=3), seed=0)
    env.reset()
    for _ in range(3):
        env.step_async(np.array([4, 4]))
        obs, reward, done, infos = env.step_wait()
    assert done.all()
    assert all("terminal_observation" in i and i["TimeLimit.truncated"] for i in infos)
    assert obs["grid"].shape == (2, 4, 17, 17)


def test_stats_helpers():
    mean, lo, hi = bootstrap_ci([1.0] * 10)
    assert mean == lo == hi == 1.0
    assert verdict([1.0, 0.9, 0.95, 1.0]) == "pass"
    assert verdict([0.05, -0.05, 0.0, 0.02]) == "fail"
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}
