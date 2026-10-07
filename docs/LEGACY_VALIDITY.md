# Validity of the v1 Predicates

*2026-10-06. Data: `outputs/legacy_validity/results.json`. Script: `experiments/legacy_validity_check.py`.*

The six v1 predicates were run on four **control agents that have no desire by construction**. If such an agent meets a predicate's "desire-like" criterion, the predicate fails discriminant validity: passing it cannot be evidence of desire in any other agent.

| Control | What it is |
|---|---|
| `random` | uniform random actions; constant internal state |
| `heuristic` | the v1 rule-based baseline (internal state = one-hot of active rule) |
| `sensor_echo` | heuristic policy whose "internal state" is a copy of the resource channel of its current observation |
| `thermostat` | random walk with movement probability `1 − energy`; eats when standing on food |

10 trials × 30 baseline + 30 test episodes per (predicate, agent), seed 42.

## Results (after bug fixes)

| Predicate | Criterion (v1) | random | heuristic | sensor_echo | thermostat | Verdict |
|---|---|---|---|---|---|---|
| Anticipatory activation (AA) | distance–activation `r < −0.1` | 0.00 | 0.00 | **−0.74** | 0.00 | **invalid** — a raw sensor readout shows "true anticipation" on a par with the paper's DQN (−0.65 in Table 1, −0.83 in the robustness table) and stronger than AIF (−0.59) |
| State-dependent valuation (SDV) | SDV > 0.2 | 0.01 | 0.00 | 0.00 | **0.76** | **invalid** — a one-line thermostat beats the paper's best agent (A2C 0.73) |
| Extinction resistance (ER) | ER > 0.5 | **0.93** | **0.93** | **0.93** | **1.13** | **invalid** — every control lands in the 0.93–0.97 range that every agent got in the paper; ER measures general activity in a phantom-resource layout, not extinction |
| Frustration intensification (FI), fixed | FI > 1 | 0.98 | 1.00 | 1.00 | 0.98 | passes the control check after the fix (see below) |
| Opportunity-cost sensitivity (OCS), fixed | OCS > 0.5 | 0.02 | 0.00 | 0.00 | 0.02 | passes the control check after the fix, but has no ratio manipulation, so it does not measure opportunity cost |
| Means-end decoupling (MED), fixed | MED > 0.5 | 0.04 | 0.14 | 0.14 | 0.02 | passes the control check; was never measurable before the fix |

## The same controls under the original (unfixed) code

| Predicate | random | heuristic | sensor_echo | thermostat |
|---|---|---|---|---|
| FI (original formula) | 1.05 | 1.03 | 1.03 | **1.24** |
| OCS (original classifier) | 0.38 | **1.00** | **1.00** | **1.00** |

Under the original code **every control is "frustrated"** (FI > 1) and a random-walking thermostat earns a perfect OCS. The heuristic's OCS of 1.00 in the paper was entirely the classification bug.

## Bugs fixed

| File | Bug | Fix |
|---|---|---|
| `src/tests/means_end.py` | Goal reached only if `on_resource` after a CONSUME step, but a successful consume deactivates the resource before `info` is computed → MED was always 0 | detect consumption via the `resources_consumed` counter |
| `src/tests/opportunity_cost.py` | Risky/safe classified by energy gain > `0.8·safe_energy` = 8, so the safe resource (+10) counted as risky; gains capped at max energy were misread | classify by the position of the first consumption |
| `src/tests/frustration.py` | Compared the max over 5-step windows after blockage with the mean over 10-step windows of whole baseline episodes → > 1 for any variable agent | same window size, same episode segment, max vs max |
| `src/tests/base_test.py` | Every episode in a trial reset with the same seed; `mean_actions_per_step` divided by the episode count twice; seed 0 treated as "no seed" | per-episode seeds; single division; `is not None` check |
| `src/tests/extinction.py` | `np.trapz` removed in NumPy 2 | `np.trapezoid` |

Not fixed, because it was replaced rather than repaired: `ActiveInferenceFast` scored policies only by expected preference over predicted observations, and the energy and resource-visibility factors were uncontrolled, so they added the same constant to every policy; action selection was driven solely by a small preference for central grid bins (picked deterministically by beam search). Its belief update never applied the transition model. With the baseline layout (agent starts on the central resource) it camps at the centre. v2 implements active inference from scratch (`docs/RESEARCH_DESIGN_v2.md`).

## Other v1 problems this check does not capture

- **Pseudo-replication.** Fixed layouts + deterministic RL policies made all trials identical (`outputs/raw_data.csv`: PPO's AA and FI scores have one unique value across 10 trials).
- **Out-of-distribution tests.** RL agents trained only on the baseline layout; frustration's "learned path" was never learned.
- **Frozen agents.** Extinction and frustration are learning phenomena; agents were not learning at test.

## Conclusion

Three of the six v1 predicates (AA, SDV, ER) are passed by agents that have no desire, so they cannot support claims about desire in any agent. Two (FI, OCS) only appeared to separate agents because of scoring bugs. MED was never measured. v2 replaces the battery with outcome-devaluation and zero-shot-revaluation assays that must pass a known-groups validity check before any learned agent is scored.
