# Wanting vs. Habit — v2 Results

*Status: complete (2026-10-10).*

All numbers are means over 10 independently trained seeds, with 95% percentile-bootstrap CIs over seeds. Verdicts follow the pre-registered rules in `docs/RESEARCH_DESIGN_v2.md` §6, as amended:
- Amendment 2: the specificity guard.
- Amendments 3–4: the competence rule.
- Amendment 5: the 30M extension.

Generated tables: `outputs/v2/results_tables.md`. Per-seed data: `outputs/v2/planners/`, `outputs/v2/model_free_assays/` (10M) and `outputs/v2/model_free_assays_30M/`.

## Summary

**Question.** What does an artificial agent need in order to *want* an outcome, valuing it by what it currently needs, rather than act from habit? Two tests from animal-learning research were validated on agents whose answer is known in advance:
- **Devaluation:** satiate one nutrient and see whether the agent switches.
- **Zero-shot revaluation:** create a need never experienced and see whether the agent immediately seeks the one item that fixes it.

**Findings.**

1. **The tests work, and the zero-shot test needs a guard.** The ideal agent passes both tests and every no-desire control fails them. Without a specificity guard, though, an agent that simply tries the unfamiliar item whenever it is needy passes zero-shot revaluation perfectly. A pilot active-inference agent did exactly that while knowing nothing about the item.
2. **Zero-shot revaluation appeared only where value is computed at decision time from a learned model of outcomes and an innate need function.** That covers the homeostatic planner and active inference: both pass both tests in every seed, and they are indistinguishable from each other.
3. **Learned value fails out of distribution, even with the right inputs and a good model.**
   - The model-based agent with a learned *reward* model passes devaluation and fails revaluation (RI +0.15).
   - Model-free agents trained to 30M steps forage as well as the oracle and pass devaluation in every seed. The recurrent agent is at ceiling (DI +1.00), with graded, near-oracle sensitivity to familiar needs. Neither revalues (RI 0.00). The recurrent agent chose C in 0 of 1,985 choices when deprived of it. The feedforward agent chose C 4 times in 1,969, fewer than the 7 times it chose C when not deprived.
   - Responding to needs, even in a graded way, in familiar need states is not the same as wanting outcomes.
4. **Devaluation alone cannot separate these.** Feedforward PPO passes devaluation with a satiety veto: it avoids what it is full of but is nearly indifferent between slightly- and badly-needed nutrients (45% vs the oracle's 100%). Recurrent PPO passes with graded need sensitivity. Both fail revaluation identically. An agent trained in a single need state is a pure habit agent (DI 0) despite perfect foraging.
5. **Framework labels don't matter; ingredients do.** Active inference and the homeostatic planner behave identically, and the novelty term had no measurable effect.

**Implications.**
- **AI welfare.** Behavioral evidence of "wanting" in an AI system should come from out-of-distribution revaluation tests with specificity controls, not from in-distribution sensitivity to state. The latter is passed by agents whose valuations are learned habits.
- **Alignment.** These results show the goal-misgeneralization pattern under controlled conditions: competent, need-sensitive-looking behavior that does not track the underlying goal off-distribution.

**What did not go to plan.**
- The novelty prediction (P8) failed.
- At the pre-registered 10M budget, the model-free agents fell short of the competence criterion. Their verdicts at 10M are "not interpretable", and one disclosed extension (Amendment 5) was needed.
- Requiring competence of the blind control was a design error, and it leaves P2 formally uninterpretable.

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
| `rppo_free_intero` | 1.00 / 0.99 | **+1.00** [+0.99, +1.00] | **+0.00** [+0.00, +0.00] | −0.00 [−0.00, +0.00] | 0.103 | **P4 supported** |

**P3 and P4 are both supported.** At 30M both agents:
- meet the competence criterion in every seed;
- forage about as well as the oracle (mean drive 0.108 and 0.103 vs 0.100; some recurrent seeds beat it, at 0.086, because the oracle plans only 20 steps ahead);
- pass devaluation in every seed. The recurrent agent is at ceiling: DI 0.98–1.00, with the sated item chosen in 0.2% of probes.

Neither revalues. When deprived of C, `rppo_free_intero` chose C in 0 of 1,985 choices, and `ppo_free_intero` in 4 of 1,969 (0.2%, against 7 when not deprived). Both chose on 98–99% of probes. Tripling training made both agents better at everything the assays test *in distribution* and left zero-shot revaluation exactly where it started (`outputs/v2/figures/training_amount.png`).

The 30M need sweep (`outputs/v2/figures/need_sweep_30M.png`) shows that the two architectures learned different things about A:
- **Feedforward PPO learned a satiety veto.** It almost never picks a sated item (P(A) = 0.05 at level 0.5). But when A is far more depleted than the alternative (0.05 vs 0.3), it still picks A in only 45% of probes, where the oracle picks it in 100%. "Don't eat what you are full of" passes devaluation without graded valuation, and T1 cannot tell the two apart.
- **Recurrent PPO learned graded need sensitivity.** P(A) goes 0.00 → 0.14 → 0.55 (at equal need) → 0.74 → 0.81 → 0.84 as A is depleted, close to the oracle's step function. Its C curve is flat at 0.00.

The recurrent agent is therefore the strongest test of the central hypothesis. On the nutrients it was trained with, its behavior is close to indistinguishable from an ideal wanting agent. On a need it has not experienced, it is a pure habit agent.

### 3.3 What the model-free agents learned (exploratory)

*This subsection describes the 10M agents; §3.2 covers 30M.*

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

All are recorded with dates in the design doc's amendments. Amendments 1–3 were made before the results they could affect. Amendments 4 and 5 were made after some model-free results had been seen, and they say exactly which.

1. Novelty zero at test; indices conditional on a choice; exploratory T2b (Amendment 1).
2. Setpoint 0.7 → 0.5, because C's effect was censored at the ceiling and unlearnable (Amendment 2).
3. Specificity guard `RI_spec` and the `need_novelty` control added (Amendment 2).
4. Model-free architecture and training settings fixed on pilot seeds by competence only (Amendment 3).
5. How the competence criterion is applied: per agent type, on seed means. This matched code committed before any result. It was written down after two interim recurrent-PPO seeds had been seen (Amendment 4).
6. One extension of the two interoceptive model-free agents from 10M to 30M steps, triggered by the competence shortfall, with a commitment to no further extensions. It was decided after the 10M indices had been seen (Amendment 5). The 10M verdicts ("not interpretable") are reported alongside, not replaced.

## 5. Things that went wrong and were caught

- **v1 battery** (`docs/LEGACY_VALIDITY.md`): three predicates were passed by no-desire controls, two were artifacts of scoring bugs, and one was never measured.
- **An active-inference agent "passed" zero-shot revaluation out of ignorance.** In the first pilot, a flat Dirichlet prior over C's (unobservable) effect plus the KL-risk term made the uncertain item look attractive under deprivation (RI 0.77). This is a general validity threat: under deprivation, uncertainty-seeking mimics wanting.
- **The scorecard initially marked P8 "supported"** on a raw difference of 109 vs 107. It now uses the planned permutation test, as §7 requires.
- **The blind control was held to the competence criterion.** An agent that cannot sense its needs cannot know which item it needs, so its low choice rate was predictable. The rule made P2 uninterpretable even though the blind agent's DI ≈ 0 is guaranteed by construction. Competence gates should apply only to agents for which competence is achievable.
- **The 10M training budget was too small for the model-free agents.** Pilot runs at 3M had already shown the risk (Amendment 3). In hindsight, the budget should have been set by a competence target, not by a fixed step count.

---

## Postscript, 2026-10-20: what the v3 experiment changed

This document is the record of the pre-registered v2 results and is not rewritten. The project's current position is in `docs/CLAIMS.md`. In brief:

- **Finding 3 ("learned value fails out of distribution, even with the right inputs and a good model") was too strong as a general claim.** The v2 learners had one input unit per nutrient, the never-needed nutrient's input never varied, and nothing linked item C to nutrient C. No learner could have passed. v3 (`docs/HELDOUT_DESIGN.md`, `docs/RESULTS_HELDOUT.md`) gave model-free learners object-based inputs, seven nutrients, and a built-in item↔need link. By the pre-registered per-seed criterion, 4 of 20 seeds then revalued a never-needed nutrient, and only when that specific need was low. The arm-level verdict was *inconclusive*; on average the learner still preferred a less-needed familiar item.
- **Finding 2 (planners pass) and finding 5 (framework labels don't matter) stand**, with the caveat that the planners are the assays' positive controls and their passing is close to guaranteed by construction.
- **Findings 1 and 4 (the tests need guards; devaluation alone cannot separate wanting from a satiety veto) were strengthened.** v3 added directional and seeking guards and three further impostor controls, all sorted correctly.
- **The "not interpretable" competence gating used here was dropped in v3** in favor of one unconditional symmetric rule, after the project owner objected that gating lets a reader suspect selective interpretation. The v2 10M verdicts remain as reported above.
