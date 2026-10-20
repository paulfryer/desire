# Wanting vs. Habit in Artificial Agents

Which computational ingredients make an agent *want* an outcome, valuing it by what it currently needs, rather than repeat a habit? And can we tell from behavior alone, using tests checked against agents whose answer we already know?

This repository borrows two tests from animal-learning research:

- **Outcome devaluation** (Dickinson & Balleine): satiate the agent on one outcome, then see whether it switches, without any chance to relearn.
- **Zero-shot revaluation** (Robinson & Berridge 2013): put the agent in a need state it has never experienced, and see whether it immediately goes for the one item that fixes it.

Both tests are validated on scripted agents with known answers, including agents built to *fake* wanting, before any learned agent is scored. They are then run on model-free RL (feedforward and recurrent, with and without interoception), model-based RL with a learned reward model, a homeostatic planner, active inference, and, in the latest experiment, model-free learners given a built-in link between items and the needs they fix.

**The current position, with confidence levels and what is left to test: [`docs/CLAIMS.md`](docs/CLAIMS.md).** The short version follows.

## What we have learned

**Solid (methodological).**
- Behavioral tests for wanting have to be validated against impostors. The zero-shot test is passed by novelty-seeking, by a satiety veto, and by attraction to unusual states unless it carries three guards (specific, directional, seeking). With the guards, all seven scripted impostors are sorted correctly.
- Competent, graded, in-distribution need sensitivity is not evidence of wanting. A recurrent model-free agent that forages like the oracle and passes devaluation at ceiling never once seeks the remedy for an unfamiliar need (0 of 1,985 choices).

**One pre-registered experiment, primary result inconclusive.**
- Given a built-in item↔need link, model-free RL *sometimes* revalues a never-needed outcome: 4 of 20 seeds (6–44%) by the pre-set criterion, and only when that specific need was low. On average it does not: held-out revaluation +0.28 [+0.13, +0.44] against +0.96 on familiar needs, and the deprived agent still preferred a less-needed familiar item.
- Without the built-in link, or with only two trained needs, nothing transferred. The generic MLP never learned the task, so that question is untested.
- Our own pre-registered forecast (45% on clean transfer) scored worse than a uniform guess.

**Hypothesis, untested.** The same agents revalue a never-*seen* item far more than the familiar-but-never-needed one. What may block transfer is a learned aversion that a new need fails to override, which is exactly what Berridge's sodium-depleted rats do override.

**How significant.** Modest. The durable part is the methodology and the cautionary result. The positive result is real but small and in a toy world. See `docs/CLAIMS.md` §4–5 for what we are not claiming and what would change our minds.

## How our thinking evolved

| | Question | Outcome |
|---|---|---|
| **v1** (Jan–Oct 2026) | Do active-inference agents show desire signatures that RL agents don't? Six predicates, five agents. | **Retracted.** The predicates were passed by agents with no desire, or depended on scoring bugs. Kept in [`paper/legacy_v1/`](paper/legacy_v1/RETRACTED.md) as a record; see `docs/LEGACY_VALIDITY.md`. |
| **v2** (Oct 6–10) | Which ingredients produce wanting vs habit, measured by validated devaluation and revaluation assays? | Planners with an innate need function pass; learned valuation fails zero-shot revaluation. But the learners' failure was close to guaranteed by construction. `docs/RESULTS_v2.md`. |
| **v3** (Oct 10–11) | Give learning its best shot: can a learner with a built-in item↔need link revalue a need it has never had? Pre-registered, numeric priors, one scoring rule. | **Inconclusive** by the frozen rule; sometimes yes, not on average. `docs/RESULTS_HELDOUT.md`. |

## Documents

| File | Contents |
|---|---|
| `docs/CLAIMS.md` | **Start here.** Current claims with confidence levels, what we are not claiming, significance, what is left to test |
| `docs/HELDOUT_DESIGN.md` | v3 frozen pre-registration: numeric priors, the single scoring rule, outcome table, review log |
| `docs/RESULTS_HELDOUT.md` | v3 results by that rule, prior scorecard, every deviation, independent review log |
| `docs/RESEARCH_DESIGN_v2.md` | v2 design and pre-registered predictions, with dated amendments |
| `docs/RESULTS_v2.md` | v2 results against those predictions (with a postscript on what v3 changed) |
| `docs/report/` | Self-contained HTML reports of v2 and v3 (open locally; no network) |
| `docs/LITERATURE.md` | Literature review and novelty assessment |
| `docs/LEGACY_VALIDITY.md` | The audit that retired v1 |
| `paper/legacy_v1/` | The retracted v1 paper draft, with retraction notice |

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy scipy pandas matplotlib gymnasium torch \
    stable-baselines3 sb3-contrib scikit-learn pytest ruff pypdf
.venv/bin/python -m pytest tests
```

## Reproducing

v2:

```bash
# Planners and ground-truth controls: train (planners), then run every assay. ~15 min.
python experiments/run_planners.py --seeds 0-9

# Model-free agents: 4 agents x 10 seeds x 10M steps, then assays on final models and
# checkpoints (Amendment 3 settings). About 5 hours on a 10-core laptop.
python experiments/launch_model_free.py --seeds 0-9 --parallel 8 --steps 10000000 \
    --features split --norm-reward --n-envs 64 --n-steps 256 --batch-size 4096

# Amendment 5 extension: the two interoceptive agents continued to 30M steps.
python experiments/launch_model_free.py --agents ppo_free_intero rppo_free_intero \
    --seeds 0-9 --parallel 10 --steps 30000000 --features split --norm-reward --n-envs 64 \
    --n-steps 256 --batch-size 4096 --resume-from outputs/v2/model_free \
    --out outputs/v2/model_free_30M --assay-out outputs/v2/model_free_assays_30M

# Tables, prediction scorecard and figures -> outputs/v2/; need-sweep curves; HTML report
python experiments/analyze_v2.py
python experiments/need_sweep.py --seeds 0-9 && python experiments/need_sweep.py --plot
python experiments/build_report_v2.py
```

v3, held-out need:

```bash
python experiments/run_heldout.py validity                     # scripted controls vs the scoring rule
python experiments/launch_heldout.py main --steps 30000000 --parallel 9 \
    --learning-rate learned-level-K6=1e-4 mlp-K6=1e-4         # 70 runs, ~17 h on a 10-core laptop
for k in 0 1 2 3 4 5 6 7; do python experiments/run_heldout.py learned --shard $k/8 & done; wait
python experiments/analyze_v3.py && python experiments/plot_v3.py && python experiments/build_report_v3.py
```

Set `OMP_NUM_THREADS=1` when running training jobs in parallel. Each run records its exact settings in its `run.json`. Every v3 training and evaluation command is logged in `outputs/v3/COMMANDS.log`.

## Layout

```
src/
  env.py              NutrientGrid (v2): batched homeostatic gridworld, 3 nutrients
  env_n.py            MultiNutrientGrid (v3): 7 nutrients, object-based observations, held-out nutrient
  assays.py           v2: T0 competence, T1 devaluation, T2 zero-shot revaluation (+ specificity guard)
  assays_n.py         v3: trained and held-out revaluation with specificity, directional and seeking guards
  stats.py            v2: bootstrap CIs, permutation tests, Holm, verdict rule
  scoring_v3.py       v3: the single frozen scoring rule, t-intervals, sign-flip tests, Brier
  agents/
    controls.py       v2 known-answer controls
    planners.py       homeostatic planner / oracle, learned-reward model-based, active inference
    model_free.py     stable-baselines3 PPO / recurrent PPO adapters
    multi.py          v3 scripted impostors and the relational / MLP policy networks
experiments/          runners, launchers, frozen analyses, exploratory plots, report builders
tests/                unit tests (assay validity on controls, scoring rule, analysis end to end)
outputs/v2/, v3/      per-seed assay results (JSON), summaries, tables, figures; models and logs are git-ignored
docs/                 claims, designs, results, literature, HTML reports
paper/legacy_v1/      the retracted v1 paper
```
