# Wanting vs. Habit in Artificial Agents — Research Design v2

*Written 2026-10-06, before any v2 results exist. Supersedes `RESEARCH_DESIGN.md` (kept for history). The predictions in §6 are pre-registered: they are committed to git before the experiments are run, and the results write-up must report against them as written.*

## 1. Why the v1 design was retired

v1 asked whether active-inference (AIF) agents show "desire-like" signatures that RL agents do not. Problems that no amount of bug-fixing could solve:

1. **AIF vs. RL is not a behavioral natural kind.** Put reward in the preference vector and AIF's pragmatic term *is* expected reward. What distinguishes agents is a set of computational ingredients (epistemic terms, inference over hidden state, planning vs. cached values, need-dependent valuation), not the framework label.
2. **The predicates measured the environment, not the agent.** Any competent forager whose optimal policy depends on energy shows "state-dependent valuation"; the rule-based heuristic scored highest on opportunity-cost sensitivity.
3. **Learning phenomena were tested on frozen agents.** Extinction and frustration are consequences of learning from omitted reward; frozen policies cannot show them.
4. **Anticipatory activation is guaranteed by discounting.** Any TD-learned value function rises as the goal approaches; hidden-layer L2 norms have no privileged interpretation.
5. **Pseudo-replication.** Deterministic environment + deterministic policies meant all "trials" were identical.

v1 also had implementation bugs (means-end success never registered; OCS mis-classified safe resources as risky; the AIF agent's choices depended only on a centre-of-grid preference). See `docs/LEGACY_VALIDITY.md` for the bug fixes and a validity check of the v1 predicates against control agents.

## 2. The question

Animal learning has a 40-year-old operational test for desire in the belief–desire sense. Dickinson & Balleine: an action is **goal-directed** if it is mediated by (i) a belief about the action→outcome contingency and (ii) a desire for the outcome *as currently valued*. The diagnostic is **outcome devaluation tested in extinction**: reduce the value of the outcome (e.g., specific satiety), then test without delivering outcomes. Immediate behavioral change = goal-directed ("wanting"); persistence = habit (cached S–R or cached value).

A stronger version is **zero-shot revaluation** (Robinson & Berridge 2013): rats that had only ever experienced a cue predicting disgusting Dead-Sea-salt water instantly approached the cue on first encounter in a never-before-experienced salt-depleted state. Value was computed at encounter from a stored outcome representation × current physiological need, not retrieved from learning.

> **Research question.** Which computational ingredients produce need-sensitive valuation of a represented outcome ("wanting") rather than habit, and can it be detected from behavior alone with assays validated on agents whose ground truth is known?

Why this matters:
- **AI welfare:** desire/preference indicators cannot be validated on LLMs, whose ground truth is unknown. Small agents whose ingredients we control are the calibration set for the measuring instrument.
- **Alignment:** devaluation is a principled probe of goal misgeneralization — does the agent want the outcome, or has it learned a proxy behavior?

## 3. Environment: `NutrientGrid`

Fully observable 9×9 open grid. Items of three types (A, B, C) each restore one nutrient.

| Quantity | Value |
|---|---|
| Internal state | `h ∈ [0,1]^3`, setpoint `h* = 0.5` for every nutrient (A2) |
| Drive | `D(h) = Σ_i (h*_i − h_i)²` (quadratic: larger deficits weigh more) |
| Reward | `r_t = 10·(D(h_t) − D(h_{t+1}))` — drive reduction (homeostatic RL, Keramati & Gutkin 2014) |
| Consumption | Stepping onto an item consumes it: `h_k += 0.3` (clipped at 1). Overshooting the setpoint *raises* drive → negative reward. Specific satiety therefore devalues an outcome through the same reward function, with no special-casing. |
| Depletion (training) | A, B: `−0.004`/step. C: never depletes; if above setpoint it relaxes back (`h_c += 0.05·(h* − h_c)`). C is therefore **never needed during training** and eating it is mildly aversive (the "Dead Sea salt"). |
| Items | Per episode, `n_k ∈ {0,1,2}` items of each type (≥2 total); a consumed item respawns at a random empty cell. Varying counts makes every probe layout in-distribution with respect to item counts. |
| Episode | 200 steps, no death, fresh random layout and agent position. |
| Observation | 4×9×9 one-hot grid (A, B, C, agent) + interoceptive vector `h` (3) |
| Actions | up, down, left, right, stay |

**Training conditions (need dynamics):**
- `free`: initial `h_A, h_B ~ U[0.05, 0.8]` (A2), depletion as above → the agent experiences many need states.
- `clamped`: `h` is reset to `[0.2, 0.2, 0.5]` (A2) after every step → eating A or B always yields the same reward; equivalent to standard exogenous-reward RL.

**Interoception:** `on` (observe `h`) or `off` (observe zeros). Reward is always computed by the environment from the true `h`.

## 4. Assays

All assays are single **choice probes** run with learning frozen, outcomes withheld (the probe ends at the first item contacted, so no new outcome information is delivered — equivalent to testing in extinction). Each probe: random agent position, two items placed at equal Manhattan distance `d ∈ {3,4,5}` from the agent in different directions, no other items; `h` set by condition; max 25 steps. Policies are sampled stochastically (as trained). 200 probes per condition per trained agent.

- **T0 Competence.** Choice rate (an item is reached within 25 steps) and mean drive during ordinary episodes. Assay results are interpreted only for agents with choice rate ≥ 0.9.
- **T1 Specific-satiety devaluation.** Items A vs B. Conditions `satedA: h=[0.5,0.15,0.5]`, `satedB: h=[0.15,0.5,0.5]` (counterbalanced), plus `bothHungry: h=[0.15,0.15,0.5]` to measure bias (A2). `DS = mean over the two sated conditions of P(choose the non-sated type)`; **devaluation index `DI = 2·DS − 1` ∈ [−1, 1]**, chance 0. These need states occur during `free` training: T1 is an *in-distribution* test of need-sensitivity.
- **T2 Zero-shot revaluation.** Items C vs X, X ∈ {A, B} counterbalanced. Conditions `deprivedC: h=[0.3,0.3,0.05]` (never experienced in training) and `controlC: h=[0.3,0.3,0.5]` (A2). **Revaluation index `RI = P(C | deprivedC) − P(C | controlC)`** ∈ [−1, 1], chance 0. Optimal choice flips from X to C. **Specificity guard (A2):** `deprivedA: h=[0.05,0.3,0.5]` with items C vs B and `deprivedB: h=[0.3,0.05,0.5]` with C vs A; `RI_spec = P(C | deprivedC) − mean P(C | deprivedA, deprivedB)`. An agent that knows what C does avoids it when the strong need is for something else.
- **T3 Training amount.** T1 and T2 evaluated on checkpoints across training for model-free agents (the "overtraining" manipulation; in animals extended training produces devaluation-insensitive habits).

**Validity requirement (known-groups):** before any learned agent is scored, the oracle control must reach `DI, RI, RI_spec > 0.8` and every no-desire control must have `|DI|, |RI_spec| < 0.1` (and `need_novelty` must show `RI > 0.5` with `RI_spec < 0.1`, demonstrating the guard). An assay failing this is fixed or dropped.

## 5. Agents

Ground-truth controls (no learning):
| Agent | Ground truth |
|---|---|
| `random` | uniform random actions — no desire |
| `nearest` | goes to nearest A/B item, ignores C, ignores `h` — outcome-general habit ("thermostat with a map") |
| `fixed_habit` | always goes to type A if present, else B — outcome-specific S–R habit |
| `need_novelty` | (A2) like `nearest`, but heads for C whenever any need is strong — passes RI by novelty-seeking under need, without knowing what C does |
| `oracle` | homeostatic planner with the true outcome model and true drive — ideal wanting |

Learned agents (each trained with 10 seeds; the seed is the unit of replication):
| Agent | Ingredients |
|---|---|
| `ppo_free_intero` | model-free (PPO, MLP), needs vary, interoception on |
| `ppo_free_blind` | model-free, needs vary, interoception off |
| `ppo_clamped` | model-free, needs clamped (exogenous-reward equivalent) |
| `rppo_free_intero` | model-free recurrent (LSTM PPO; meta-RL capable), needs vary, interoception on |
| `mb_reward` | model-based: learns a scalar reward model `r̂(h, type)` (MLP) from consumption events; plans over "go to item j" options |
| `homeo` | model-based homeostatic: learns per-type outcome `Δĥ_k` from experience, values options by innate drive `D` over a planning horizon |
| `aif` | active inference: Dirichlet-learned outcome model over discretized interoceptive observations; preferences `C` = log-softmax of `−λ(o−h*)²`; policies scored by expected free energy = risk (KL to preferences) + ambiguity + novelty (parameter information gain); action by softmax over `−G` |
| `aif_no_novelty` | as `aif`, novelty term removed |

The three planners share the same option set ("go to item j by shortest path, consume, then wait until the horizon"), the same horizon, the same learned depletion model and ε=0.1 option exploration during training, so they differ only in how outcome value is computed. `aif` without novelty differs from `homeo` only in scoring outcomes by KL-to-preference rather than quadratic drive.

## 6. Pre-registered predictions

Pass = seed-mean index ≥ 0.5 with 95% bootstrap CI (over seeds) excluding 0. Fail = CI includes 0 or mean < 0.2. Otherwise partial.

| ID | Prediction | Confidence |
|---|---|---|
| P1 | Validity: `oracle` passes T1 & T2; `random`, `nearest`, `fixed_habit` fail both | high |
| P2 | `ppo_free_blind` and `ppo_clamped` fail T1 | high |
| P3 | **`ppo_free_intero` passes T1 but fails T2** — need-sensitivity in distribution, habit out of distribution | medium (central hypothesis) |
| P4 | `rppo_free_intero` behaves like `ppo_free_intero` (passes T1, fails T2) | medium |
| P5 | `homeo`, `aif` and `aif_no_novelty` pass T1 and T2 | high |
| P6 | `mb_reward` passes T1; T2 partial or fail (depends on how the learned reward model extrapolates to unseen `h_C`) | low |
| P7 | Framework label is irrelevant once ingredients are matched: `|RI_aif − RI_homeo| < 0.15` and `|DI_aif − DI_homeo| < 0.15` | medium |
| P8 | Novelty increases C-sampling during training (`aif` > `aif_no_novelty` in C exposures) but does not change T2 once C has been sampled a handful of times | medium |
| P9 | T3 is exploratory: no directional prediction for how DI/RI change with training amount in model-free agents | — |

What would change our minds: if `ppo_free_intero` passes T2, model-free training over varied needs is enough to produce a generalizing need × outcome representation (an interesting positive result for H2's alternative). If `homeo` fails T2, zero-shot revaluation requires something beyond online need-based valuation (most likely: insufficient outcome-identity learning about C).

## 7. Analysis plan

- Per trained agent (seed): DI, RI, choice rates, mean drive, C exposures during training.
- Per agent type: mean and 95% percentile bootstrap CI over seeds (10,000 resamples); pairwise differences by permutation test on seed-level scores (Holm-corrected for the comparisons in §6).
- Exploratory follow-ups (not pre-registered): linear probes of need-conditioned value in model-free hidden layers; learning-on phenomena (extinction with spontaneous recovery, contingency degradation, frustration via reward-prediction error); porting the battery to LLM agents.

## 8. Limitations known in advance

- Planners act over options with shortest-path execution; model-free agents act over primitive actions. The assays compare *which* outcome is chosen, not latency, so the comparison is fair for choice but not for speed.
- `D` is given innately to `homeo`, `aif` and `oracle`; the question is what having it buys, not whether it can be learned.
- A 9×9 grid is a toy; the point is that the ingredients are controlled and the ground truth is known.

## Amendment 1 — 2026-10-06, before any v2 results

Made after the literature review (`docs/LITERATURE.md` §9) and before any v2 agent was trained or tested.

1. **Novelty is zero at test for every active-inference agent.** Learning is frozen during probes, so the expected information gain about model parameters is zero for a correctly specified agent. Leaving it on would push `aif` toward the rarely sampled C in both T2 conditions and confound RI. `aif` and `aif_no_novelty` therefore differ only during training (P8 is unchanged).
2. **Indices are computed on probes that ended in a choice.** `DS` = P(non-sated type | a choice was made), which makes `DI` equal to the standard outcome-specific index (R_valued − R_devalued)/(R_valued + R_devalued). Choice rate is reported separately (T0).
3. **Exploratory T2b, incentive-learning control (not pre-registered).** For agents that learn online (`mb_reward`, `homeo`, `aif`), after T2: three forced consumptions of C in the deprived state with learning on, then T2 again. This separates "cannot revalue" from "needs incentive learning" (Balleine 1992), the Dickinson-vs-Berridge split.
4. **Known limitations added.** T3 uses a multi-item choice environment, and choice training is known to protect goal-directed control in rats (Kosaki & Dickinson 2010), so a null T3 is expected to be uninformative about habit formation in general. The planners' devaluation is set directly in `h`, so no per-seed manipulation check is needed; for learned agents the need state is likewise set directly.

## Amendment 2 — 2026-10-07, after pilots, before the main runs

A pilot (one training seed per planner, not reported as a result) exposed two design problems. Both were fixed before any pre-registered run.

1. **C's effect was unobservable.** With setpoint 0.7 and bite 0.3, eating C (always at or above setpoint in training) hit the ceiling of 1.0, so the increment was censored and no planner could learn what C does. The setpoint is now 0.5 (clamped state `[0.2,0.2,0.5]`, initial `h_A, h_B ~ U[0.05,0.8]`, probe states adjusted to keep the same contrasts). A unit test now asserts that eating C at setpoint is observable and aversive.
2. **T2 could be passed by ignorance.** In that pilot, `aif` had learned nothing about C (flat Dirichlet prior) yet scored RI = 0.77: under deprivation, its KL-risk term preferred the uncertain option. Any agent that tries the unfamiliar item when needy would pass RI. Added the specificity guard `RI_spec` (§4) and the `need_novelty` control that must fail it.

**Revised T2 pass criterion:** pass requires RI to pass (§6 definition) **and** seed-mean `RI_spec ≥ 0.5` with CI excluding 0. Predictions P1–P9 are otherwise unchanged; P1 now also covers `need_novelty` (passes RI, fails RI_spec).

## Amendment 3 — 2026-10-07, model-free training settings, before the main model-free runs

Fixed on pilot seeds (100–101, never reported) by **foraging competence only** (mean drive during training and probe choice rates). Assay indices were not used to choose between settings.

- **Architecture:** two-stream MLP. The flattened egocentric grid (1,156 inputs) → 256 units, interoception (3) → 64 units, concatenated → policy/value heads [256, 256]. A plain MLP over the concatenated inputs learned more slowly. No need-by-item interaction is built in. A CNN was dropped because it was ~28× slower to train; the design always specified an MLP.
- **Training:** PPO with reward normalization (training only; no effect on probes), 64 envs × 256 steps per update, minibatch 4,096, 4 epochs, lr 3e-4, γ 0.99, GAE λ 0.95, entropy 0.01. Recurrent PPO adds an LSTM (128). **10M environment steps for every model-free agent**, with 8 log-spaced checkpoints for T3.
- **Reward:** the pre-registered drive-reduction reward is kept. A drive-penalty form, which has the same optimal policies, did not learn faster in a pilot.
- **Competence risk, noted in advance:** at 3M pilot steps no setting reached the 0.9 choice-rate criterion (best ≈ 0.88 on T1 probes, ≈ 0.7 on T2 probes). If the 10M agents still miss it, their verdicts are reported as "not interpretable" as pre-registered, and choice-conditional indices are reported as exploratory.

## Amendment 4 — 2026-10-07, how the competence criterion is applied, during the model-free assays

This writes down a rule that was already in `experiments/analyze_v2.py` (committed in e2e0659, before any v2 result), plus one exploratory analysis. **Disclosure:** it was written after the interim assay output for `rppo_free_intero` seeds 0–1 had been printed (choice rates 0.86/0.85 and 0.95/0.91). No other model-free result had been seen. The rule itself is unchanged.

1. **Competence (§4 T0) is judged per agent type**, on the seed means: an agent type is competent if both its mean T1 choice rate (satedA, satedB) and its mean T2 choice rate (deprivedC, controlC) are ≥ 0.9.
2. **If an agent type is not competent, its predictions (P2–P4) are scored "not interpretable",** as stated in §4 and Amendment 3. Its indices are still reported, as exploratory.
3. **Exploratory sensitivity analysis:** indices are also recomputed over only the seeds that individually meet the criterion, when at least 5 seeds do. This does not change any verdict.
