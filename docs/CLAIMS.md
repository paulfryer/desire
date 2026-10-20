# What this project claims, and how confident it is

*Last revised 2026-10-20, after the v3 held-out-need experiment. This is the single place where the project's current position is stated. The experiment documents (`RESULTS_v2.md`, `RESULTS_HELDOUT.md`) are records of pre-registered results and are not rewritten when our thinking changes; this file is.*

## 1. The question, and how it changed

**v1 (Jan–Oct 2026, retracted).** Do active-inference agents show behavioral signatures of desire that RL agents don't? Six predicates, five agent types. Retracted because the predicates were never validated: three were passed by agents with no desire by construction, two worked only because of scoring bugs, one was never measured. `paper/legacy_v1/RETRACTED.md`, `docs/LEGACY_VALIDITY.md`.

**v2 (Oct 6–10).** Reframed: which computational ingredients make an agent *want* an outcome (value it by what it currently needs) rather than act from habit? Two assays from animal learning, outcome devaluation and Berridge-style zero-shot revaluation, validated first on known-answer agents, then run on planners and model-free RL. `docs/RESEARCH_DESIGN_v2.md`, `docs/RESULTS_v2.md`.

**v3 (Oct 10–11).** v2's learners failed zero-shot revaluation, but their failure was close to guaranteed by construction: nothing linked an item to the need it fixes. v3 asked whether learning *can* produce wanting for a never-experienced need when the learner is given the minimal innate structure an animal plausibly has. Pre-registered with numeric priors and one scoring rule. `docs/HELDOUT_DESIGN.md`, `docs/RESULTS_HELDOUT.md`.

## 2. Claims, by confidence

### High confidence: methodological

**C1. Behavioral tests for "wanting" must be validated against agents built to fake it, and the zero-shot revaluation test needs three guards.** Without them, each of these passes the raw test: trying the unfamiliar item whenever needy (novelty), avoiding only what one is full of (satiety veto), going for whatever looks unusual, and breaking down randomly on unfamiliar input. With the guards (specific to the deprived need, directional, actively seeking), all seven scripted impostors are sorted correctly in every world tested.
*Evidence:* `outputs/v3/validity/`, RESULTS_HELDOUT §7; v2's pilot active-inference agent that "passed" out of ignorance (RESULTS_v2 §"What did not go to plan").
*Would be overturned by:* an impostor strategy that passes all three guards. We have not found one; we have not proved none exists.

**C2. In-distribution need sensitivity is not evidence of wanting, even when it is graded and near-ideal.** A recurrent model-free agent trained to 30M steps forages as well as the oracle, passes devaluation at ceiling (DI 1.00), tracks familiar needs in a graded way, and never once seeks the remedy for an unfamiliar need (0 of 1,985 choices). The v3 best-shot learner scores 0.96 on familiar needs and 0.28 on the never-needed one.
*Evidence:* RESULTS_v2 §3, RESULTS_HELDOUT §1.
*Scope:* a toy grid world and PPO/recurrent PPO. The direction of the claim (sensitivity ≠ wanting) does not depend on scale; the specific numbers do.

**C3. Agents that compute value at decision time from a learned outcome model and an innate need function pass both tests; the homeostatic planner and active inference are behaviorally indistinguishable here.** Framework labels did not matter; ingredients did. The epistemic/novelty term of active inference had no measurable effect.
*Evidence:* RESULTS_v2 §2, §5.
*Scope:* these are the positive controls for the assays, and the result is close to guaranteed by their construction. It is a validity result, not a discovery.

### Medium confidence: one pre-registered experiment, inconclusive primary

**C4. With a built-in item↔need link, model-free RL can sometimes produce zero-shot revaluation for a never-needed outcome, but not reliably and not on average.** By the pre-registered per-seed criterion, 4 of 20 seeds transferred (Clopper–Pearson 6–44%); those seeds chose the held-out item only when its specific need was low, not when a different need was low or when the item's state was merely unusual. The arm-level verdict is *inconclusive* (RI_H +0.28 [+0.13, +0.44]). On average the deprived agent still preferred a less-needed familiar item (SI_H −0.43 [−0.74, −0.12]).
*What this does to v2's conclusion:* "learned valuation does not generalize to unexperienced needs" is contradicted as a universal. As a description of the typical learner it held up better than my own pre-registered prior, which put 45% on clean transfer and scored worse than a uniform guess (Brier 1.01 vs 0.75).
*Would be strengthened by:* more seeds under the same frozen rule. *Would be weakened by:* a rerun with fixed world codes and varied network init showing the 4/20 are a property of particular worlds rather than of learning.

**C5. Removing the built-in link, or training on only two needs, removes transfer entirely.** Both arms were *negligible* by the frozen rule; no seed in either ever chose the held-out item when deprived. Two caveats stated in the results: both arms learned the familiar needs less well (0.63 and 0.75 vs 0.96), and the learned-link arm used a lower learning rate and was still improving when the budget ended. Their best-learning seeds (familiar ≈ 0.9) still scored 0, so weaker learning is unlikely to be the whole story, but the confound is not removed.
*Evidence:* RESULTS_HELDOUT §3–4.

**C6. Not detected (n = 10 each): whether a nutrient-general value read-out matters, and whether experiencing the held-out nutrient's full side matters.** Both comparisons came back with Holm p = 1.0. With ten seeds a moderate effect would often be missed. We claim nothing either way.

**C7. Whether a generic network (plain MLP) can do any of this was not tested.** The MLP never learned the familiar needs (RI_T 0.00) at the fixed budget, so there was nothing to transfer. This is a failure of our design for that arm, not evidence about generic learners.

### Hypothesis: not yet tested

**H1. What blocks transfer is a learned, outcome-specific aversion, not a failure to generalize need-reading.** The same agents revalue an item with a never-seen code far more than the held-out nutrient they had experienced but never needed (RI_novel +0.68 vs RI_H +0.28, paired difference +0.40 [+0.26, +0.54]). In training the held-out item always over-filled the agent, so it was mildly aversive, like salt to a rat that has never been sodium-depleted. Berridge's rats override that aversion the first time the need arises; these agents mostly do not. The novel-code probe had no specificity or directional guards, so a "go for the unfamiliar when needy" policy could produce the same numbers. This is the most interesting lead and it is untested.

## 3. What we are not claiming

- Anything about agents outside this toy world, this reward structure, or PPO.
- That generic neural networks learn wanting (C7).
- That learning cannot produce wanting (C4 contradicts it).
- That learning reliably produces wanting (C4 contradicts that too).
- That these assays are ready to apply to language models or to AI-welfare questions. They are a template for what such a test would need to include (known-answer validation, impostor controls, out-of-distribution needs), not a test that can be run on an LLM today.

## 4. How significant is this

Modest, and we think honestly so.

- The durable contribution is **methodological** (C1, C2): a worked example of validating behavioral "wanting" assays against impostors before trusting them, and a demonstration that the most natural evidence (graded, competent need-sensitivity) is passable by habits. This is relevant to anyone proposing behavioral indicators of desire or preference in AI systems, and it is the part we would defend in front of a hostile audience.
- The **positive existence result** (C4) is real but small: 4 of 20 seeds, in one toy world, with a built-in link doing some of the work. It is enough to say "sometimes", not enough to say how often or why.
- The **ingredient results** (C5–C7) are mostly negative, unresolved or confounded.
- The **hypothesis** (H1) is the most interesting thing to come out of the project and has zero direct evidence.

A paper could be written now around C1–C2 with v2 and v3 as demonstrations. A paper claiming that learning produces (or doesn't produce) wanting could not; the primary result is inconclusive and would sit in the middle of it as a hole.

## 5. What is left to test, in priority order

1. **Resolve the primary.** Add seeds to the best-shot arm under the exact frozen rule (`HELDOUT_DESIGN.md` §6). Twenty more seeds is about 5–6 hours on this machine. Adding seeds changes the pre-registered n, so it needs a new dated amendment stating the new n before the runs, and the original 20-seed verdict stays reported alongside.
2. **Test H1 directly.** (a) A training variant in which held-out items are neutral rather than aversive (no overshoot), so the need-reading and the learned aversion can be separated. (b) Add specificity and directional guard conditions to the novel-code probe. Both are cheap (one arm each, 10 seeds) and should be pre-registered together with priors.
3. **Give the learned-link and MLP arms a fair budget.** The learned-link arm was still improving at 30M; the MLP never learned at all. Until a generic learner reaches competence on familiar needs, the "does generic learning do it" question is untested (C7). A competence-targeted budget rather than a fixed step count, set in advance, would fix this.
4. **Separate world from initialization.** Run the best-shot arm with the world codes fixed and only the network seed varied, and vice versa, to see which produces the 4/20 split.
5. **A second algorithm.** Everything so far is PPO. Recurrent/meta-RL (which can look model-based) is the obvious next one, with the same impostor controls.
6. **Scale and transfer to other settings.** Only after 1–4. Any claim about language models or welfare indicators is speculative until a version of these assays can actually be run on such a system.

## 6. Where the evidence lives

| Claim | Pre-registration | Results | Data |
|---|---|---|---|
| v1 retraction | — | `docs/LEGACY_VALIDITY.md`, `paper/legacy_v1/RETRACTED.md` | `outputs/legacy_validity/` |
| C1–C3 | `docs/RESEARCH_DESIGN_v2.md` (Amendments 1–5) | `docs/RESULTS_v2.md`, `docs/report/v2_wanting_vs_habit.html` | `outputs/v2/` |
| C1, C4–C7, H1 | `docs/HELDOUT_DESIGN.md` (frozen bb2726b, Amendment 1) | `docs/RESULTS_HELDOUT.md`, `docs/report/v3_heldout_need.html` | `outputs/v3/`, `outputs/v3/COMMANDS.log` |

All experiments were run and written up by Claude (Opus 5.5 at the time), with the pre-registration reviewed independently before freezing and the v3 write-up checked independently against the raw data afterwards. The review logs are in the design and results documents.
