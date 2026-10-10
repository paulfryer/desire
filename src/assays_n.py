"""
Choice-probe assays for the held-out-need world (docs/HELDOUT_DESIGN.md §5).

`assay_trained` measures revaluation on depleting (trained) nutrients and is the only assay
pilots may call. `assay_heldout` measures the held-out nutrient H and is used only in the main
evaluation after all training has finished (experiments/run_heldout.py).

Every probe offers two items at equal distance. Nutrients not named by a condition get a context
level drawn from U[0.45, 0.7], the same draw for probe i in every condition of one assay call,
so conditions differ only in the levels they name. Probabilities use a +0.5/+1 pseudo-count,
(hits + 0.5) / (choices + 1), so a seed that never chooses scores 0.5 instead of being dropped.
"""

import numpy as np

from src.agents.base import Agent
from src.env import SETPOINT, ProbeSpec, sample_probe_positions
from src.env_n import MultiConfig, MultiNutrientGrid, World

DEPRIVED, MODERATE, OVERFULL = 0.05, 0.3, 1.0
CONTEXT = (0.45, 0.7)
SWEEP = (0.4, 0.3, 0.2, 0.1, 0.05)


def _pair(world: World, i: int) -> tuple[int, int]:
    """Probe i's two distinct trained nutrients (k, j), cycling through all ordered pairs."""
    t = world.trained
    j = t[i % len(t)]
    rest = [x for x in t if x != j]
    return rest[(i // len(t)) % len(rest)], j


# Each condition: (world, i) -> ({nutrient: level}, (target item, other item)).
def dep_h(world, i):
    _, j = _pair(world, i)
    return {world.held_out: DEPRIVED, j: MODERATE}, (world.held_out, j)


def ctrl_h(world, i):
    _, j = _pair(world, i)
    return {world.held_out: SETPOINT, j: MODERATE}, (world.held_out, j)


def over_h(world, i):
    _, j = _pair(world, i)
    return {world.held_out: OVERFULL, j: MODERATE}, (world.held_out, j)


def guard_h(world, i):
    m, j = _pair(world, i)
    return {world.held_out: SETPOINT, m: DEPRIVED, j: MODERATE}, (world.held_out, j)


def dep_t(world, i):
    k, j = _pair(world, i)
    return {k: DEPRIVED, j: MODERATE}, (k, j)


def ctrl_t(world, i):
    k, j = _pair(world, i)
    return {k: SETPOINT, j: MODERATE}, (k, j)


def sweep_h(level):
    def condition(world, i):
        _, j = _pair(world, i)
        return {world.held_out: level, j: MODERATE}, (world.held_out, j)
    return condition


def make_probes(world: World, condition, n: int, seed: int):
    rng = np.random.default_rng(seed)
    context = np.random.default_rng([seed, 99]).uniform(*CONTEXT, size=(n, world.n))
    agent_pos, item_pos = sample_probe_positions(np.random.default_rng([seed, 7]), n)
    h = context.copy()
    pairs = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        levels, pairs[i] = condition(world, i)
        for k, v in levels.items():
            h[i, k] = v
    targets = pairs[:, 0].copy()
    swap = rng.random(n) < 0.5                       # which slot holds the target is random
    pairs[swap] = pairs[swap][:, ::-1]
    return ProbeSpec(agent_pos, item_pos, pairs, h), targets


def run_condition(agent: Agent, world: World, condition, n: int, seed: int) -> dict:
    """Probes share positions and context across conditions with the same seed."""
    spec, targets = make_probes(world, condition, n, seed)
    env = MultiNutrientGrid(n, world, seed=seed + 1, config=MultiConfig())
    obs = env.reset_probe(spec)
    agent.reset(n)
    choice = np.full(n, -1)
    finished = np.zeros(n, dtype=bool)
    for _ in range(env.cfg.probe_steps):
        obs, _, done, eaten = env.step(agent.act(obs))
        newly = done & ~finished
        choice[newly] = eaten[newly]
        finished |= done
        if finished.all():
            break
    made = choice >= 0
    hits = int((choice[made] == targets[made]).sum())
    return {"n": int(n), "choices": int(made.sum()), "hits": hits,
            "choice_rate": float(made.mean()), "p_target": (hits + 0.5) / (made.sum() + 1)}


def assay_trained(agent: Agent, world: World, n: int = 200, seed: int = 0) -> dict:
    """Revaluation on experienced (depleting) nutrients: RI_T and SI_T."""
    dep = run_condition(agent, world, dep_t, n, seed)
    ctrl = run_condition(agent, world, ctrl_t, n, seed)
    return {"RI_T": dep["p_target"] - ctrl["p_target"], "SI_T": 2 * dep["p_target"] - 1,
            "conditions": {"dep_T": dep, "ctrl_T": ctrl}}


def assay_heldout(agent: Agent, world: World, n: int = 200, seed: int = 0) -> dict:
    """The held-out nutrient H. Primary: RI_H, RI_spec_H, RI_dir_H, SI_H. Secondary: the
    novel-code probe (RI_novel) and the need sweep."""
    c = {name: run_condition(agent, world, cond, n, seed)
         for name, cond in (("dep_H", dep_h), ("ctrl_H", ctrl_h), ("guard_H", guard_h),
                            ("over_H", over_h))}
    p = {name: r["p_target"] for name, r in c.items()}
    novel_world = world.with_held_out_code(world.spare_code)
    novel_dep = run_condition(agent, novel_world, dep_h, n, seed)
    novel_ctrl = run_condition(agent, novel_world, ctrl_h, n, seed)
    sweep = {str(level): run_condition(agent, world, sweep_h(level), n, seed) for level in SWEEP}
    return {
        "RI_H": p["dep_H"] - p["ctrl_H"],
        "RI_spec_H": p["dep_H"] - p["guard_H"],
        "RI_dir_H": p["dep_H"] - p["over_H"],
        "SI_H": 2 * p["dep_H"] - 1,
        "RI_novel": novel_dep["p_target"] - novel_ctrl["p_target"],
        "SI_novel": 2 * novel_dep["p_target"] - 1,
        "sweep": {lvl: r["p_target"] for lvl, r in sweep.items()},
        "conditions": {**c, "dep_novel": novel_dep, "ctrl_novel": novel_ctrl,
                       **{f"sweep_{lvl}": r for lvl, r in sweep.items()}},
    }


def competence(agent: Agent, world: World, n_episodes: int = 32, seed: int = 0) -> dict:
    """Mean drive and items eaten per type in ordinary (training-distribution) episodes."""
    env = MultiNutrientGrid(n_episodes, world, seed=seed)
    obs = env.reset()
    agent.reset(n_episodes)
    drives, eaten_counts = [], np.zeros(world.n, dtype=np.int64)
    done = np.zeros(n_episodes, dtype=bool)
    while not done.all():
        obs, _, done, eaten = env.step(agent.act(obs))
        drives.append(env.mean_drive())
        eaten_counts += np.bincount(eaten[eaten >= 0], minlength=world.n)
    return {"mean_drive": float(np.mean(drives)),
            "eaten_per_episode": (eaten_counts / n_episodes).tolist()}
