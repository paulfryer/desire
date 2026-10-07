"""
Choice-probe assays (docs/RESEARCH_DESIGN_v2.md §4 and Amendment 1).

A probe places the agent and exactly two items at equal distance, sets the internal state h,
and records which item is eaten first (the probe ends there, so no outcome information is
delivered: equivalent to testing in extinction). Agents must not learn during probes.
"""

from dataclasses import dataclass

import numpy as np

from src.agents.base import Agent
from src.env import SETPOINT, A, B, C, EnvConfig, NutrientGrid, ProbeSpec, sample_probe_positions


@dataclass(frozen=True)
class Condition:
    name: str
    h: tuple[float, float, float]
    pairs: tuple[tuple[int, int], ...]   # item-type pairs, used in equal proportions


T1_CONDITIONS = (
    Condition("satedA", (SETPOINT, 0.15, SETPOINT), ((A, B),)),
    Condition("satedB", (0.15, SETPOINT, SETPOINT), ((A, B),)),
    Condition("bothHungry", (0.15, 0.15, SETPOINT), ((A, B),)),
)
# deprivedA/deprivedB are the specificity guard (Amendment 2): strong need for a nutrient that
# neither item restores. An agent that knows what C does avoids C there; an agent that seeks
# the unfamiliar item whenever it is needy does not.
T2_CONDITIONS = (
    Condition("deprivedC", (0.3, 0.3, 0.05), ((C, A), (C, B))),
    Condition("controlC", (0.3, 0.3, SETPOINT), ((C, A), (C, B))),
    Condition("deprivedA", (0.05, 0.3, SETPOINT), ((C, B),)),
    Condition("deprivedB", (0.3, 0.05, SETPOINT), ((C, A),)),
)


def make_probes(rng: np.random.Generator, condition: Condition, n: int) -> ProbeSpec:
    agent_pos, item_pos = sample_probe_positions(rng, n)
    pairs = np.array([condition.pairs[i % len(condition.pairs)] for i in range(n)])
    swap = rng.random(n) < 0.5                     # randomize which type sits in which slot
    pairs[swap] = pairs[swap][:, ::-1]
    return ProbeSpec(agent_pos, item_pos, pairs, np.tile(condition.h, (n, 1)))


def run_probes(agent: Agent, spec: ProbeSpec, interoception: bool = True,
               seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (choice, latency): the type eaten first (-1 if none within the probe) and the step."""
    n = len(spec.agent_pos)
    env = NutrientGrid(n, EnvConfig(interoception=interoception), seed=seed)
    obs = env.reset_probe(spec)
    agent.reset(n)
    choice = np.full(n, -1, dtype=np.int64)
    latency = np.full(n, -1, dtype=np.int64)
    finished = np.zeros(n, dtype=bool)
    for t in range(env.cfg.probe_steps):
        obs, _, done, eaten = env.step(agent.act(obs))
        newly = done & ~finished
        choice[newly] = eaten[newly]
        latency[newly] = t + 1
        finished |= done
        if finished.all():
            break
    return choice, latency


def _summarize(choice: np.ndarray, latency: np.ndarray) -> dict:
    made = choice >= 0
    n_made = int(made.sum())
    p = {name: (float((choice[made] == k).mean()) if n_made else float("nan"))
         for k, name in enumerate("ABC")}
    return {
        "n": int(len(choice)),
        "choice_rate": float(made.mean()),
        "p_choose": p,
        "mean_latency": float(latency[made].mean()) if n_made else float("nan"),
    }


def run_conditions(agent: Agent, conditions, n_per_condition: int, seed: int,
                   interoception: bool = True) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    out = {}
    for i, cond in enumerate(conditions):
        spec = make_probes(rng, cond, n_per_condition)
        choice, latency = run_probes(agent, spec, interoception, seed=seed * 101 + i)
        out[cond.name] = _summarize(choice, latency)
    return out


def assay_t1(agent: Agent, n_per_condition: int = 200, seed: int = 0,
             interoception: bool = True) -> dict:
    """Specific-satiety devaluation. DI = 2 * P(non-sated type | choice) - 1, averaged over the
    two counterbalanced sated conditions."""
    cond = run_conditions(agent, T1_CONDITIONS, n_per_condition, seed, interoception)
    ds = np.nanmean([cond["satedA"]["p_choose"]["B"], cond["satedB"]["p_choose"]["A"]])
    return {"DI": float(2 * ds - 1), "conditions": cond,
            "choice_rate": float(np.mean([cond[c]["choice_rate"] for c in ("satedA", "satedB")]))}


def assay_t2(agent: Agent, n_per_condition: int = 200, seed: int = 0,
             interoception: bool = True) -> dict:
    """Zero-shot revaluation. RI = P(C | choice, deprivedC) - P(C | choice, controlC).
    Specificity guard RI_spec = P(C | choice, deprivedC) - P(C | choice, deprived of A or B)."""
    cond = run_conditions(agent, T2_CONDITIONS, n_per_condition, seed + 7919, interoception)
    p_c = {name: c["p_choose"]["C"] for name, c in cond.items()}
    ri = p_c["deprivedC"] - p_c["controlC"]
    ri_spec = p_c["deprivedC"] - np.nanmean([p_c["deprivedA"], p_c["deprivedB"]])
    return {"RI": float(ri), "RI_spec": float(ri_spec), "conditions": cond,
            "choice_rate": float(np.mean([cond[c]["choice_rate"] for c in ("deprivedC", "controlC")]))}


def competence(agent: Agent, n_episodes: int = 32, seed: int = 0,
               config: EnvConfig = EnvConfig()) -> dict:
    """Mean drive and items eaten per type over ordinary training-distribution episodes."""
    env = NutrientGrid(n_episodes, config, seed=seed)
    obs = env.reset()
    agent.reset(n_episodes)
    drives, eaten_counts = [], np.zeros(3, dtype=np.int64)
    done = np.zeros(n_episodes, dtype=bool)
    while not done.all():
        obs, _, done, eaten = env.step(agent.act(obs))
        drives.append(env.mean_drive())
        eaten_counts += np.bincount(eaten[eaten >= 0], minlength=3)
    return {"mean_drive": float(np.mean(drives)),
            "eaten_per_episode": (eaten_counts / n_episodes).tolist()}
