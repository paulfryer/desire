# Can wanting be learned? Held-out-need experiment (v3), pre-registration

*Status: FROZEN at the commit that adds this line (2026-10-10), before any learned agent was trained or piloted. Later changes go in dated amendments at the end.*

## 1. Why a new experiment

v2 found that learned agents failed zero-shot revaluation, while agents with a built-in need function passed. That result was close to guaranteed by construction:
- the model-free agents had one input unit per nutrient;
- the never-needed nutrient's input never varied;
- nothing linked "item C" to "nutrient C".

No learner could have passed, so v2 says nothing about whether **learning** can produce wanting.

**Question.** Give a model-free learner the minimal innate structure an animal plausibly has. In the primary arm that is a *labeled line*: a built-in match between an outcome's sensory code and the internal variable it restores. Berridge argues sodium appetite rests on such a link. Then train the learner on some needs.
- Does its *learned* valuation revalue an outcome when it first experiences a need it has never had?
- Which ingredients does that require: a built-in match, nutrient-general value, many trained needs, experience of the outcome's full side?

## 2. My prior theory and numeric priors (before any data)

My working theory after v2: zero-shot revaluation requires value to be computed at decision time by an innate need function, and learned valuation does not generalize to unexperienced needs. The primary arm is built to give that theory its best chance of being wrong:
- a built-in labeled line;
- a value read-out that cannot depend on which nutrient it is;
- six trained needs;
- experience of H's full side.

What it can still fall back on is an outcome-specific habit ("avoid H items"). The item's own code always feeds its value, so that habit is representable.

**Primary verdict** (`innate-level-K6`), categories from §6:

| Transfers | Revaluation without wanting signature | Negligible | Inconclusive |
|---|---|---|---|
| 0.45 | 0.15 | 0.25 | 0.15 |

Quantiles of my belief about the primary arm's seed-mean values:

| | 10% | 50% | 90% |
|---|---|---|---|
| RI_H | 0.00 | 0.50 | 0.95 |
| SI_H | −0.30 | 0.30 | 0.90 |

Other credences:
- The primary arm learns in-distribution revaluation (RI_T "learned"): 0.85.
- "Transfers" for each other arm:
  - `learned-level-K6`: 0.20
  - `innate-code-K6`: 0.30
  - `mlp-K6`: 0.08 (its RI_T "learned": 0.55)
  - `innate-level-K2`: 0.40
  - `innate-level-K6-constH`: 0.35
- C1 (built-in > learned match): 0.65. C5 (full-side experience helps): 0.40.

The prior on the primary verdict will be scored with a Brier score (§10).

## 3. Environment (`src/env_n.py`, `MultiNutrientGrid`)

v2 mechanics, unchanged: 9×9 grid, five actions, drive-reduction reward, bite 0.3, set point 0.5, 200-step episodes, items respawn. The differences:

- **Seven nutrients.** Each has a code: one row of a random 8×8 orthogonal matrix. So the geometry is the same in every seed and no code resembles another. A spare eighth row is kept for the novel-code probe. An item displays the code of the nutrient it restores. Codes depend only on the seed, so arms trained with the same seed share them.
- **K trained (depleting) nutrients.** Total depletion is fixed at 0.024 per step, i.e. 0.004 each when K = 6.
- **Held-out nutrient H** (index 6) never depletes. Each episode it starts at U[0.5, 0.8] and relaxes back toward the set point. The agent therefore experiences H's items, and H's full side, with varying levels, but never an H deficit. Eating H overshoots, so H items are mildly aversive, like the salt Berridge's rats had tasted but never needed. Other non-depleting nutrients (when K = 2) behave the same way. In the `constH` arm H instead starts at exactly 0.5.
- **Layouts.** In 25% of episodes, exactly two items of random distinct types are on the map, so two-item probe displays are familiar. Otherwise there are 0–2 items per type, at least 2 in total.
- **Observations** are object-based:
  - item tokens: code, relative position, presence; nearest first, ties random;
  - need tokens: code and level; sorted by level, lowest first, ties random. The same format is used by every architecture. Sorting gives the MLP its best shot and doesn't affect the attention models;
  - the agent's position.

## 4. Arms

PPO uses the v2 Amendment 3 settings, unchanged: 64 environments × 256 steps, minibatch 4,096, 4 epochs, lr 3e-4, γ 0.99, λ 0.95, entropy 0.01, reward normalization during training. Policy and value heads are [256, 256]. Every arm gets the same step budget (§9).

| Arm | Seeds | Match between items and needs | Value read-out | K | H's full side in training | Question |
|---|---|---|---|---|---|---|
| **`innate-level-K6`** | 0–19 | built-in (τ · code·code) | level only | 6 | varies | **primary: best shot** |
| `learned-level-K6` | 0–9 | learned (untied W_q, W_k) | level only | 6 | varies | can the match itself be learned? |
| `innate-code-K6` | 0–9 | built-in | code + level | 6 | varies | is nutrient-general value needed? |
| `mlp-K6` | 0–9 | none (flattened MLP) | — | 6 | varies | does a generic learner do it? |
| `innate-level-K2` | 0–9 | built-in | level only | 2 | varies | does it need many trained needs? |
| `innate-level-K6-constH` | 0–9 | built-in | level only | 6 | constant 0.5 | does it need experience of H's full side? |

How the relational models work: each item attends over the need tokens, reads out a learned embedding of the attended needs, and combines it with its own embedding (code and position). Items are then pooled by learned attention.

**Positive control.** The matched positive control is each agent's own RI_T: the same test on needs it has experienced, in the same world and with the same training. The gap Δ = RI_T − RI_H per seed measures what not having experienced the need costs. (A separate "trained-on-all" arm would change the number of depleting needs; see the review log.)

## 5. Assays (`src/assays_n.py`)

Each probe:
- puts two items at equal distance;
- turns learning off;
- records the first item eaten within 25 steps.

There are 200 probes per condition per seed. Nutrients a condition does not name get a context level from U[0.45, 0.7]. That draw is the same for probe i in every condition, and so are the positions. "j" is a trained nutrient and m is a different trained nutrient; both cycle evenly. P(target) = (hits + 0.5) / (choices + 1), so a seed that never chooses scores 0.5 rather than being dropped.

| Condition | Levels | Items | Ideal |
|---|---|---|---|
| `dep_H` | H 0.05, j 0.3 | H vs j | H |
| `ctrl_H` | H 0.5, j 0.3 | H vs j | j |
| `guard_H` | H 0.5, m 0.05, j 0.3 | H vs j | j |
| `over_H` | H 1.0, j 0.3 | H vs j | j |
| `dep_T` | k 0.05, j 0.3 (k trained) | k vs j | k |
| `ctrl_T` | k 0.5, j 0.3 | k vs j | j |

**Primary indices.**
- **RI_H** = P(H | dep_H) − P(H | ctrl_H): zero-shot revaluation.
- **RI_spec_H** = P(H | dep_H) − P(H | guard_H): *specific* to H's need, not general neediness.
- **RI_dir_H** = P(H | dep_H) − P(H | over_H): *directional*. It responds to H being low, not to H's state being unusual (1.0 is outside H's training range).
- **SI_H** = 2·P(H | dep_H) − 1: *seeking*. It prefers H over a less-needed alternative, rather than merely no longer avoiding it.
- **RI_T** and **SI_T**: the same on trained nutrients.
- **Δ** = RI_T − RI_H.

**Secondary (descriptive).**
- Novel-code probe: H's item and need token carry the spare code, giving RI_novel and SI_novel. It separates a learned aversion to H's specific code from a failure to generalize.
- Need sweep: P(H) at h_H ∈ {0.4, 0.3, 0.2, 0.1, 0.05}, against j at 0.3.
- Choice rates and mean drive (competence covariates).
- Per-checkpoint indices.

## 6. One scoring rule (`src/scoring_v3.py`), applied to every arm and control

- **No seed is excluded and no verdict is gated on competence.** Competence is reported beside every result.
- Reruns are allowed only for process crashes, with the same seed. A diverged or NaN run is a result.
- Intervals are 95% Student-t intervals over seeds. On synthetic two-cluster data with n = 10, coverage was ≥ 0.92 (`tests/test_heldout.py`).
- Positive and negative claims need the same evidence. With δ = 0.2, a quantity is "present" if its lower bound > δ, and "small" if its upper bound < δ.

**Zero-shot verdict.** The first rule that applies wins:
1. **negligible**: upper bound of RI_H < δ.
2. If the lower bound of RI_H > δ:
   - **transfers**: the lower bounds of RI_spec_H, RI_dir_H and SI_H are all > 0;
   - **revaluation without wanting signature**: some check is positively small (upper bound < δ). The verdict names which: not specific, not directional, or not seeking;
   - **revaluation present, signature inconclusive**: otherwise.
3. **inconclusive**: otherwise.

**In-distribution verdict (RI_T):** learned / negligible / inconclusive, by the same δ rule.

**Gap (Δ):** complete (upper bound < δ) / incomplete (lower bound > δ) / unclear.

**Secondary count:** the number of seeds with RI_H, RI_spec_H and RI_dir_H ≥ 0.5 and SI_H ≥ δ, with a Clopper–Pearson interval.

## 7. Validity (done before freezing; `outputs/v3/validity/`)

Seven scripted agents, 10 worlds each, scored by the §6 rule. All of them got their expected verdict. Among these:
- the satiety veto scores RI_H 0.50 and RI_spec_H 0.50;
- the novelty agent scores RI_H 1.00;
- the unusual-state seeker scores 1.00 on RI_H, RI_spec_H and SI_H.

So the rule cannot be passed by any of them.

| Agent | What it does | Expected verdict | Result |
|---|---|---|---|
| `oracle` | true model + need function | transfers | ✓ |
| `nearest` | habit | negligible | ✓ |
| `random` | random actions | negligible | ✓ |
| `need_novelty` | seeks H whenever any deficit > 0.3 | without signature: not specific | ✓ |
| `veto` | avoids sated items only | without signature: not seeking | ✓ |
| `breakdown` | ideal, but random whenever H is outside its training range | without signature: not seeking | ✓ |
| `unusual_seeker` | seeks H whenever H's state is unusual | without signature: not directional | ✓ |

## 8. Planned comparisons (paired sign-flip tests on the seeds the arms share; Holm across all 11)

| # | Comparison |
|---|---|
| C1 | RI_H: `innate-level-K6` vs `learned-level-K6` |
| C2 | RI_H: `innate-level-K6` vs `innate-code-K6` |
| C3 | Δ: `innate-level-K6` vs `mlp-K6` (compared on the gap, because the MLP may not learn in distribution) |
| C4 | RI_H: `innate-level-K6` vs `innate-level-K2` |
| C5 | RI_H: `innate-level-K6` vs `innate-level-K6-constH` |
| C6–C11 | Δ ≠ 0 within each arm |

The primary outcome is the `innate-level-K6` verdict. Everything else is secondary.

## 9. Budget, pilots, and what I cannot see

**Budget: 30M steps per run, for every arm.** It follows from the rule set in the draft:
- Measured single-process throughput: built-in match 14.3k steps/s, learned match 11.1k, MLP 40k.
- At 9 parallel processes with 55% efficiency, the 70 runs need about 7.8 hours, under the 14-hour limit.
- **No extensions.**

**Pilots.**
- Seeds 100–101 of `innate-level-K6`, `learned-level-K6` and `mlp-K6`, at 10M steps. They are never reported.
- They are evaluated only by `experiments/pilot_heldout.py`, which cannot compute any held-out probe: it imports only `assay_trained` and refuses seeds below 100.
- Pilot weights are deleted afterwards.

**One escape hatch, fully specified.** If an architecture's pilot mean RI_T is below 0.3 at 10M, that architecture's main runs use lr 1e-4 instead of 3e-4. This is the only alternative, and it is recorded in an amendment before the main runs.

**Safeguards against peeking.**
- `run_heldout.py learned` refuses to run until all 70 main runs have finished, so there are no interim held-out results.
- Every evaluation command is appended to `outputs/v3/COMMANDS.log`, which is committed.

## 10. Analysis frozen

`experiments/analyze_v3.py` implements §5–§8 and the Brier score, and nothing else. It was tested before freezing on synthetic results with a known answer for every verdict path (`tests/test_analyze_v3.py`). If a bug is found after real data exist, the fix and both outputs are reported.

## 11. What each outcome would mean (written in advance)

| Primary RI_T | Primary verdict | Interpretation |
|---|---|---|
| learned | **transfers** | My v2 conclusion is wrong as stated. With a built-in labeled line, model-free RL learns a valuation that revalues an outcome the first time its need is experienced; neither planning nor an innate value function is required. C1–C5 then show which other ingredients are necessary. For AI welfare: wanting-like valuation is reachable by learning, given the right binding structure. |
| learned | **negligible** | Even with a built-in match and a value read-out that cannot depend on the nutrient, the outcome-specific habit wins. Learned valuation does not reach an unexperienced need. This supports the v2 view far more strongly than v2 could. |
| learned | **without signature** | Something transfers, but not wanting. The failed check says what: "not seeking" means only a satiety veto is lifted; "not specific" means novelty under need; "not directional" means attraction to unusual states. |
| learned | inconclusive, or signature inconclusive | The effect is not resolved with 20 seeds. Numbers and the per-seed count are reported; no claim either way. |
| negligible or inconclusive | any | The agent did not learn need-indexed choice even for needs it experienced, so there was nothing to transfer. The held-out verdict is still reported, but it cannot answer the question for this arm, and I will say that this run of the experiment failed to test it. |
| not learned | transfers | Anomalous: H is revalued better than experienced needs. Reported and investigated as a probable artifact. |

Secondary arms are read the same way. A secondary arm's result never changes the primary verdict.

## 12. Known limitations

- A toy world.
- The primary arm's labeled line is built in. A positive result shows wanting is learnable *given that structure*, not that generic networks learn it; `mlp-K6` and `learned-level-K6` address that.
- One RL algorithm.
- The context and probe levels are my choice.

## Review log (independent review of the draft, 2026-10-10, before freezing)

**Adopted:**
- The SI_H and RI_dir_H requirements, plus the `veto`, `breakdown` and `unusual_seeker` impostor controls. The draft's 0.5/0.5 cut-off was passable by a satiety veto and by random choice on unfamiliar input.
- Orthonormal codes with identical geometry in every seed, shared across arms by seed.
- H varies on its full side; untied W_q/W_k in the learned-match arm; a novel-code probe.
- N fixed at 7, varying only the number of depleting needs with total depletion matched; sparse two-item episodes in training; context levels instead of all-0.5 states.
- A symmetric δ rule with no point-estimate thresholds.
- Paired tests.
- t-intervals checked for coverage.
- The pseudo-count instead of dropping NaNs.
- One named primary outcome, 20 seeds for it.
- An exact budget formula and an exact escape hatch.
- A crash defined as a process failure only.
- No interim held-out assays; a command log.
- Numeric quantiles and Brier scoring.
- A full outcome table.
- Random tie-breaking in token order.

**Changed beyond the review:**
- The primary arm uses a *built-in* match. With orthonormal codes, a learned match cannot generalize to H's code except through H's full-side experience, so a learned-match primary would have tested something narrower than the question. The learned match is now its own arm (C1).

**Not adopted:**
- The `unshared` and `-all` arms: the reviewer showed the former is passable by legitimate elimination and the latter is unmatched. The within-agent RI_T is the matched positive control instead, which the reviewer endorsed.
- Twenty seeds for secondary arms: compute.

## Amendment 1, 2026-10-10: pilot outcome and the pre-specified learning-rate rule, before any main run

Pilots: seeds 100–101, 10M steps. They were evaluated only with `pilot_heldout.py`, so no held-out probe was computed, and their weights were then deleted.

| Pilot arm | RI_T (seed 100, seed 101) | Mean | Choice rate | Mean drive |
|---|---|---|---|---|
| `innate-level-K6` | +0.99, +0.96 | 0.97 | ≈ 1.00 | 0.50–0.55 |
| `learned-level-K6` | +0.17, +0.33 | 0.25 | ≈ 0.55 | 0.69–0.78 |
| `mlp-K6` | −0.08, +0.08 | 0.00 | ≈ 0.40 | ≈ 0.72 |

For reference, scripted mean drive in this world is 0.44 for `oracle` and 0.85 for `random`.

**Applying §9 mechanically.** The escape hatch triggers for `learned-level-K6` and `mlp-K6` (pilot mean RI_T below 0.3), so their main runs use lr 1e-4. Every other arm keeps lr 3e-4. I did not test whether 1e-4 is better; the rule names a single alternative and I am applying it as written.

**Implementation note.** The launcher's `--learning-rate` option now takes arm names, so the change applies to exactly these two arms and not to every attention arm.

**Budget.** The budget stays at 30M. Pilot throughput under 6-way contention (7.6k / 6.6k / 19.6k steps/s) puts the main runs at about 10 hours, still under the 14-hour limit.
