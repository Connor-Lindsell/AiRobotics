# Greedy Basline
One-liner: At each step, the greedy baseline picks whichever valid action minimises the robot's immediate trip cost — robot → source cell → destination cell — with no lookahead.

Technical explanation:

The cost function (test.py:684) is simply compute_travel(robot_pos, source_pos, dest_pos) — the total Euclidean distance the robot travels to pick up a letter and drop it. At every step, the baseline calls action_masks() to get the legal moves, then does min(valid_actions, key=_greedy_cost) (test.py:270), selecting the cheapest single move with no awareness of what future moves that choice forces.

During training, this runs live inside GreedyCompetitionWrapper._run_greedy() (train.py:136): on each reset(), the wrapper clones the exact board state onto a shadow environment and runs the greedy solver to completion, recording _cumulative_travel. Then on the final step of the RL episode, a bonus of COMPETITION_SCALE × (greedy_travel − rl_travel) is added to the reward (train.py:176) — positive if RL beat greedy, negative if it didn't. This makes the RL agent's objective explicitly relative: it's not just "complete the task" but "complete it in fewer metres than the myopic heuristic would have taken on the same board." The greedy baseline is cheap to run because it's deterministic and has no neural network inference overhead, so the per-reset shadow rollout adds negligible wall-clock cost to training.

# Masking at Each Stage

## C1 (Stage 1) — Tight mask, 1 correct letter, no distractors
**One-liner:** Only allows moves where the source letter exactly matches the target letter for that Wordle slot, and the destination is a required empty slot.

**Technical explanation:** `action_masks()` (wordle_env.py:383) iterates every occupied cell, looks up the letter there, and only enables the action `src * N_CELLS + dst` if `dst` is in `required_slots`, is currently empty, and the letter at `src` equals `target_word[WORDLE_CELL_ID_TO_IDX[dst]]`. With only one correct letter on the board and no distractors, there is essentially one valid move at a time — the mask collapses the action space to a single correct placement per step, making the task learnable from scratch. The step limit is capped at 10 (`MAX_STEPS_PER_STAGE[1]`).

---

## C2 (Stage 2) — Tight mask, 5 correct letters + 5 distractors
**One-liner:** Same tight letter-matching mask as C1 but now five correct letters and five distractor letters share the staging area, so the agent must identify which object to move.

**Technical explanation:** The masking policy is identical to C1 — only `(src, dst)` pairs where the letter matches the target slot are unmasked. What changes is the board: five correct letters and five wrong-letter distractors are all present in staging. Because distractors can never satisfy the letter-match condition for any required slot, their source cells produce zero unmasked destinations, effectively hiding them from the action space without any extra logic. The agent must learn to ignore distractors purely through the mask structure, and the step limit rises to 15.

---

## C3 (Stage 3) — Semi-constrained mask, all Wordle slots pre-blocked with wrong letters
**One-liner:** Adds a two-rule mask: pieces sitting in Wordle slots can be evicted to any free staging cell, while pieces in staging can only move to a Wordle slot if their letter matches.

**Technical explanation:** C3 introduces a new problem: all five Wordle slots start occupied by wrong letters, so the agent must first evict those before placing correct ones. `action_masks()` branches on whether the source is a Wordle cell or a staging cell. For a Wordle source, any empty non-forbidden staging destination is legal — **except** correctly-placed letters, which are permanently locked (`wordle_correct[wi]` check prevents eviction). For a staging source, only the letter-matching tight rule from C1/C2 applies. This two-rule structure forces a two-phase behaviour — clear, then place — without the agent being able to make arbitrary or contradictory moves. Step limit is 25 to accommodate the extra clearing phase.

---

## C4 (Stage 4) — Loose mask, same board as C3 but action space opened up
**One-liner:** Removes the letter-matching constraint entirely — any piece can move to any empty non-forbidden cell — forcing the agent to learn correct sequencing from reward signal alone.

**Technical explanation:** The board setup is identical to C3, but `action_masks()` now permits any `(src, dst)` pair where `src` is occupied, `dst` is empty, `dst` is not in `FORBIDDEN_STAGING_IDS`, and `src` does not hold a correctly-placed Wordle letter. There is no longer any letter-identity check — the agent could legally move any piece anywhere. The sole hard guard remaining is that correctly-placed letters can never be evicted, preserving irreversible progress. Everything else — which piece to pick up, where to put it — must be learned from the shaped reward signal and the competition bonus against the greedy baseline. The step limit is 35, and this is the stage where the RL policy is genuinely tested against the greedy solver.

---

# Design Choices

| Design Dimension | Choice | Detail |
|---|---|---|
| **Algorithm** | MaskablePPO (sb3-contrib) | PPO with invalid-action masking; masks are applied at the logit level so illegal moves get zero probability |
| **Policy network** | MlpPolicy | Fully-connected; input → hidden layers → actor + critic heads |
| **Action space** | `Discrete(8281)` | `src_cell × 91 + dst_cell`; encodes every possible pick-and-place pair on the 91-cell grid |
| **Observation space** | `Box(0, 1, shape=(2686,))` | 2 robot pos + 91×28 cell block (occupied, letter one-hot[26], is_correct) + 5 needs-clearing flags + 5×26 target word one-hot + 1 stage indicator |
| **State representation** | Flat float32 vector | All values normalised to [0, 1]; robot xy divided by workspace extents, letters as one-hot, stage as `stage/5` |
| **Reward shaping** | Step + completion + competition bonus | Per-step travel penalty, slot-placement bonus, and `COMPETITION_SCALE × (greedy_travel − rl_travel)` on episode end |
| **Curriculum** | 4 stages (C1→C4) | Increasing board complexity and progressively looser action masks; agent promotes on success-rate threshold |
| **Action masking** | Stage-dependent (tight → loose) | C1/C2: letter-match only; C3: two-rule evict+place; C4: any legal non-forbidden move |
| **Step limits** | Per-stage caps | C1: 10, C2: 15, C3: 25, C4: 35 — episode truncates if limit exceeded |
| **Invalid action penalty** | −50.0 | Applied immediately if agent selects a masked action (guard against mask bypass) |
| **Learning rate** | 3e-4 | Adam; constant schedule |
| **n_steps / batch_size** | 4096 / 128 | 32 minibatches per PPO update |
| **clip_range** | 0.2 | Standard PPO clipping |
| **ent_coef** | 0.01 | Low entropy — policy is expected to converge; exploration via curriculum, not entropy |
| **vf_coef** | 1.0 | Boosted (default 0.5) because critic loss is in the hundreds due to competition-bonus scale |
| **gamma** | 0.99 | Standard discount |
| **Grid** | 13×7 = 91 cells | 0.75 m spacing; Wordle slots at row 3, cols 4–8; forbidden staging zone cols 3–9 × rows 0–4 |
| **Greedy baseline** | Nearest-neighbour heuristic | `min(valid_actions, key=trip_distance)` — no lookahead; run on shadow env at each `reset()` |

---

# Reward Function

| Reward Type | Amount | Trigger |
|---|---|---|
| Step penalty | −1.0 | Every step |
| Travel cost | −2.0 × metres | Every step (robot trip distance) |
| Correct placement | +20.0 | Letter placed in correct Wordle slot (once per slot) |
| Clearing bonus | +15.0 | Wrong letter evicted from a Wordle slot to staging |
| Wrong slot penalty | −20.0 | Letter placed in wrong Wordle slot |
| Evict correct letter | −10.0 | Correctly-placed letter moved out of its slot |
| Word complete | +100.0 | All required slots correctly filled |
| Greedy competition | ±10.0 × Δmetres | Episode end — positive if RL travelled less than greedy, negative if more |
| Invalid action | −50.0 | Agent selects a masked (illegal) action |

