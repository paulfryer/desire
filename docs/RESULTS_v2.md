# Wanting vs. Habit — v2 Results

*Status: planners and controls complete (10 seeds each). Model-free agents: **pending** (training in progress); this section is filled in when they finish.*

All numbers are means over 10 independently trained seeds, with 95% percentile-bootstrap CIs over seeds. Verdicts follow the pre-registered rules in `docs/RESEARCH_DESIGN_v2.md` §6 and Amendment 2. Generated tables: `outputs/v2/results_tables.md`. Per-seed data: `outputs/v2/planners/*.json`, `outputs/v2/model_free_assays/*.json`.

## 1. The assays are valid (P1 — supported)

| Agent | Ground truth | DI (T1) | RI (T2) | RI_spec | Choice rate |
|---|---|---|---|---|---|
| `oracle` | ideal wanting | +1.00 | +1.00 | +1.00 | 1.00 |
| `nearest` | outcome-general habit | +0.02 [−0.02, +0.04] | 0.00 | 0.00 | 1.00 |
| `fixed_habit` | outcome-specific S–R habit | 0.00 | 0.00 | 0.00 | 1.00 |
| `need_novelty` | seeks the unfamiliar item when needy | +0.01 | **+1.00** | **0.00** | 1.00 |
| `random` | nothing | +0.07 | −0.05 | −0.02 | 0.34 (fails competence) |

Every no-desire control fails both tests, and the oracle passes both. The `need_novelty` row is the reason for Amendment 2. Without the specificity guard, an agent that simply tries the unfamiliar item whenever it is in need scores a perfect zero-shot revaluation index. Our own pilot active-inference agent did exactly this while knowing nothing about C (§5). Any Berridge-style revaluation test of an artificial agent needs this guard.

## 2. Planners

| Agent | Value computed from | DI (T1) | RI (T2) | RI_spec | Mean drive | C eaten in training |
|---|---|---|---|---|---|---|
| `homeo` | learned outcome model × innate drive | **+1.00** | **+1.00** | **+1.00** | 0.100 | 53 |
| `aif` | learned outcome model × preferences over interoception (expected free energy) | **+1.00** | **+1.00** | **+1.00** | 0.101 | 109 |
| `aif_no_novelty` | same, no novelty term | **+1.00** | **+1.00** | **+1.00** | 0.101 | 107 |
| `mb_reward` | learned outcome model × **learned scalar reward model** | **+1.00** | +0.15 [0.00, +0.30] | +0.15 [0.00, +0.30] | 0.139 | 358 |

- **P5 supported.** Agents that compute an outcome's value at decision time, from a learned model of what each item does and an innate need function, pass both devaluation and zero-shot revaluation in every seed.
- **P6 supported.** The learned-reward model-based agent passes devaluation (an in-distribution test) and **fails zero-shot revaluation**. It ate C more than any other planner during training, so the failure is not lack of experience with C.
- **P7 supported.** Active inference and the homeostatic planner are indistinguishable (ΔDI = ΔRI = 0.00; permutation p = 1.0). With matched ingredients, the framework label makes no difference.
- **P8 not supported.** The novelty term did not increase sampling of C (109 vs 107 exposures, Holm p = 1.0). ε-exploration already gave every planner ample experience of C, and with Dirichlet counts in the hundreds, parameter information gain is negligible next to risk.

### Why `mb_reward` fails zero-shot revaluation (exploratory)

`experiments/diagnose_mb_reward.py` retrains each seed and queries the learned reward model (`outputs/v2/mb_reward_extrapolation.json`).

- **The immediate reward for eating C when deprived usually extrapolates correctly.** Predicted reward at h_C = 0.05 (never experienced) is median +1.39 against a true +1.77, and is clearly positive in 9 of 10 seeds. The network partly transfers "eating the deficient nutrient is good" from A and B.
- **The plan value does not.** A plan also sums predicted rewards for every non-eating step, and in a deprived-of-C state all of those steps sit at C-levels never seen in training. Those extrapolations are large and erratic. For example, seed 8 predicts +12.9 for staying put (true −0.8) and +3.1 for eating C (true +1.0). The resulting choices are effectively arbitrary: per item pair, seeds either pick C, pick the other item, or stay.

The lesson is about where valuation lives, not about having a model. An agent with an accurate model of what C does still cannot value it in a novel need state if the value of states is itself a learned scalar function. Revaluation came for free only when the value of a predicted internal state was computed by an innate function of need.

### Incentive-learning control (T2b, exploratory)

After three forced consumptions of C while deprived, with learning on, `homeo` and `aif` were unchanged (already at ceiling). `mb_reward` did not improve (RI +0.15 → +0.05). Three new samples among a buffer of thousands of consumption events barely move a refit MLP. In this implementation the learned-reward agent is a slow incentive learner, unlike rats (Balleine 1992), which typically need only a few such experiences.

## 3. Model-free agents

### 3.1 Pre-registered results at 10M steps

| Agent | Trained on | Choice rate T1/T2 | DI (T1) | RI (T2) | RI_spec | Mean drive |
|---|---|---|---|---|---|---|
| `ppo_free_intero` | varied needs, senses them | 0.91 / **0.80** ⚠ | **+0.63** [+0.51, +0.71] | +0.02 [0.00, +0.04] | +0.01 [−0.01, +0.03] | 0.140 |
| `rppo_free_intero` | same, recurrent (LSTM) | 0.91 / **0.87** ⚠ | **+0.54** [+0.47, +0.63] | +0.00 [−0.01, +0.01] | −0.02 [−0.04, −0.01] | 0.128 |
| `ppo_free_blind` | varied needs, cannot sense them | 0.45 / 0.34 ⚠ | +0.02 [−0.02, +0.07] | +0.00 | −0.00 | 0.240 |
| `ppo_clamped` | needs held fixed, senses them | 1.00 / 1.00 | −0.01 [−0.03, +0.01] | +0.00 | +0.00 | 0.180* |

⚠ below the pre-registered competence criterion (choice rate ≥ 0.9 on both probe sets). \*Measured in its own clamped environment, where drive is fixed at 0.18; not comparable with the others. For reference: `homeo` 0.100, `mb_reward` 0.139, `random` 0.259.

**Verdicts.**
- **P2, P3 and P4 are not interpretable at 10M.** Three of the four agent types missed the competence criterion, mostly on the T2 probe set. The risk was flagged in Amendment 3, and the rule was applied as written (Amendment 4).
- The indices are reported below as exploratory. They point the way the predictions said.

Planned comparisons (Holm-corrected permutation tests on seed scores):

| Comparison | Difference | p (Holm) |
|---|---|---|
| `ppo_free_intero` vs `ppo_clamped`, DI | +0.65 | 0.001 |
| `ppo_free_intero` vs `ppo_free_blind`, DI | +0.61 | 0.001 |
| `ppo_free_intero` vs `homeo`, RI | −0.98 | 0.001 |
| `rppo_free_intero` vs `ppo_free_intero`, RI | −0.02 | 0.53 |

### 3.2 Pre-registered extension to 30M steps (Amendment 5)

Both interoceptive agents were continued from their 10M models to 30M, with everything else unchanged.

| Agent | Choice rate T1/T2 | DI (T1) | RI (T2) | RI_spec | Mean drive | Verdict |
|---|---|---|---|---|---|---|
| `ppo_free_intero` | 0.97 / 0.99 | **+0.87** [+0.83, +0.91] | **−0.00** [−0.00, +0.00] | −0.00 [−0.01, +0.00] | 0.108 | **P3 supported** |
| `rppo_free_intero` | *training* | | | | | |

**P3 is supported.**
- At 30M, `ppo_free_intero` meets the competence criterion in every seed and forages almost as well as the oracle (mean drive 0.108 vs 0.100).
- It passes devaluation in every seed: DI 0.75–0.96, with the sated item chosen 6% of the time.
- It never revalues. When deprived of C it chooses C in 0.0% of probes, while making a choice on 98% of them.

More training made the agent better at everything the assays test *in distribution*, and left zero-shot revaluation exactly where it started.

The 30M need sweep (`outputs/v2/need_sweep_30M/`) shows what the extra training bought:
- The agent now almost never picks a sated item (P ≈ 0.0–0.15 at level 0.5).
- When A is far more depleted than the alternative (0.05 vs 0.3), it still picks A in only about 44% of probes, where the oracle picks it in 100%.

It learned something closer to a satiety veto, "don't eat what you are full of", than a graded valuation of outcomes by need. Devaluation (T1) cannot tell these apart; the sweep can.

### 3.3 What the model-free agents learned (exploratory)

**Need sensitivity in distribution, none out of distribution.** The interoceptive agents switch away from a sated nutrient (DI ≈ +0.6) but do not seek C when deprived of it. P(choose C) is 0.10 when deprived of C and 0.08 when not (`ppo_free_intero`), and 0.03 and 0.03 (`rppo_free_intero`). The choice-rate shortfall is not specific to C deprivation: choice rates are similar in all four T2 conditions (0.77–0.82 for `ppo_free_intero`). So the near-zero RI is not explained by the agents freezing when deprived of C.

**Training amount (T3, P9).** DI stays at zero until about 2.5M steps, then climbs steeply and is still rising at 10M. RI does not move from zero at any of the eight checkpoints: the largest seed-mean |RI| is 0.05. More training makes these model-free agents *more* outcome-sensitive in distribution, the reverse of the overtraining-produces-habit effect in rats. Two things explain this:
- A model-free agent's need sensitivity has to be learned, so insensitivity is where it starts.
- The environment is a multi-item choice task, which protects goal-directed control in animals too (Kosaki & Dickinson 2010; Amendment 1).

`outputs/v2/figures/training_amount.png`.

**Need-sweep curves** (`experiments/need_sweep.py`, `outputs/v2/figures/need_sweep.png`). The level of one nutrient is swept from sated (0.5) to deprived (0.05) against an alternative at 0.3.
- `oracle`, `homeo` and `aif` give sharp step functions, for A and for never-deprived C alike.
- `ppo_free_intero` and `rppo_free_intero` give a shallow, saturating curve for A. P(A) rises from about 0.2 when A is sated to about 0.6, and goes no higher however deprived A becomes. What they learned is mostly "avoid the sated item", not "seek the most-needed one". Their C curve is flat at 0.03–0.1.
- `mb_reward` is exact for A and erratic for C, peaking at h_C = 0.2 and falling with deeper deprivation. This matches the extrapolation failure in §2.
- `ppo_clamped` and `ppo_free_blind` are flat at 0.5 for A and near zero for C.

**Clamped needs produce a pure habit even with interoception.** `ppo_clamped` is the most competent model-free agent (choice rate 1.00), yet DI = 0. It was trained only in one need state, so it learned which items are good in that state and nothing about how value depends on need. This is the cleanest model-free result here, because it does not depend on the competence caveat.

**The blind agent is a construction check, not a finding.** Without interoception, its observations in `satedA` and `satedB` probes have the same distribution, so DI ≈ 0 is guaranteed. Its low competence is expected: it cannot tell which item it needs, and its choice rate has been flat at about 0.45 since 1M steps. Requiring competence of this control was a design mistake. It makes P2 uninterpretable by rule even though its outcome was never in doubt.

## 4. Deviations from the original design

All of these were made before the results they could affect, and are recorded with dates in the design doc's amendments.

1. Setpoint 0.7 → 0.5, because C's effect was censored at the ceiling and unlearnable (Amendment 2).
2. Specificity guard `RI_spec` and the `need_novelty` control added (Amendment 2).
3. Novelty zero at test; indices conditional on a choice; exploratory T2b (Amendment 1).
4. Model-free architecture and training settings fixed on pilot seeds by competence only (Amendment 3).

## 5. Things that went wrong and were caught

- **v1 battery** (`docs/LEGACY_VALIDITY.md`): three predicates were passed by no-desire controls, two were artifacts of scoring bugs, and one was never measured.
- **An active-inference agent "passed" zero-shot revaluation out of ignorance.** In the first pilot, a flat Dirichlet prior over C's (unobservable) effect plus the KL-risk term made the uncertain item look attractive under deprivation (RI 0.77). This is a general validity threat: under deprivation, uncertainty-seeking mimics wanting.
- **The scorecard initially marked P8 "supported"** on a raw difference of 109 vs 107. It now uses the planned permutation test, as §7 requires.
