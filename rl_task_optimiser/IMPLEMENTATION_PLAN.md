# RL Task Sequencer — Implementation Plan

## Overview

Two-model strategy: deploy a reliable masked model for simulation validation first,
then produce a fully-learned model for real-robot deployment. Clean stage gates
between each phase mean there is always a working fallback.

---

## Phase 1 — Sim Deployment (Masked Model)

**Goal:** Validate the full ROS2 integration end-to-end in simulation.

**Why masked model first:**
Perception is simulated and error-free, so the mask's oracle letter-check is
safe. The model is reliable and already validated. This phase proves the
integration plumbing works before introducing any model uncertainty.

**What to do:**
- Deploy `wordle_ppo_latest.zip` (current C3/C4 trained checkpoint)
- Build the ROS2 node that subclasses `TaskSequencerEvaluator` from `test.py`
- ROS2 node receives target word via subscriber, letter block positions from
  perception topic, calls `build_env → run_episode → get_task_sequence`
- Run simulation tests using `python test.py` to confirm policy behaviour
  before wiring into ROS2

**Success criteria:**
- ROS2 node receives a word, produces a valid pick/place sequence
- Simulated robot executes the sequence without errors
- RL beats or matches the greedy baseline on C3 in sim (consistent with
  current ~65% beat rate)

**Model used:** `models/wordle_ppo_latest.zip` — do not modify during this phase.

---

## Phase 2 — Retrain with Loose Masking

**Goal:** Produce a model that genuinely learns letter discrimination, not one
that relies on the mask's oracle letter-check.

**The masking problem:**
In C1/C2/C3, `action_masks()` contains this check:

```python
if letter == self.target_word[WORDLE_CELL_ID_TO_IDX[dst]]:
```

This pre-solves letter-to-slot assignment. The model only learns ordering and
travel distance — it never has to identify which letter belongs where. In the
real robot, if perception misreads a letter, the mask silently misdirects and
the model has no ability to recover.

**What "loose masking" means:**
Do NOT remove `MaskablePPO` — with 8,281 possible actions, unmasked training
is impractical. Instead, apply C4-style masking throughout all curriculum stages.
C4's mask enforces physical constraints only:

| Constraint | C1/C2/C3 mask | C4 mask (target) |
|---|---|---|
| Can't pick from empty cell | ✓ | ✓ |
| Can't place into occupied cell | ✓ | ✓ |
| Can't evict correctly-placed letter | ✓ | ✓ |
| Can't place in forbidden zone | ✓ | ✓ |
| Letter must match target slot | ✓ oracle | ✗ model must learn |

**How to implement in `train.py`:**
Add a `MASK_MODE` flag:

```python
MASK_MODE = "tight"   # current behaviour — C1/C2/C3 use oracle letter check
MASK_MODE = "loose"   # C4-style masking for all stages
```

Pass `MASK_MODE` into `WordleEnv` so `action_masks()` uses the right policy.

**Starting point:**
Continue from `wordle_ppo_latest.zip` — do NOT retrain from scratch. The model
already has spatial navigation and travel optimisation in its weights. It only
needs to learn letter discrimination. Expected convergence: 300–500k additional
steps (vs 1.2M from scratch).

**Curriculum order:** C1 → C2 → C3 → C4, all with loose masking.

**Important:** Keep `wordle_ppo_latest.zip` frozen. Train the new model to a
separate checkpoint name (e.g. `wordle_ppo_loose_latest.zip`) so Phase 1
deployment is never disrupted.

---

## Phase 3 — Sim Validation of Retrained Model

**Goal:** Confirm the loosely-masked model matches or beats the original before
touching real hardware.

**What to do:**
- Run `python test.py` with the new checkpoint loaded
- Compare C1/C2/C3 beat rates against the masked model baseline
- If retrained model matches or beats → promote to real robot deployment
- If it underperforms → continue training; masked model stays on real robot

**Success criteria:**
- C3 beat rate ≥ current masked model (~65%)
- Word completion success rate does not regress on C1/C2
- Model handles novel dictionary words (test on `HOLDOUT_WORD = GREAT`) at
  comparable success rate to training words

---

## Phase 4 — Real Robot Deployment

**Goal:** Deploy the loosely-masked model to the physical robot.

**Why it's safer than the masked model on real hardware:**
The loosely-masked model learned letter discrimination in its weights. If
perception misidentifies a block, the model has seen ambiguous letter states
during training and has learned to handle them. The masked model would silently
accept a wrong perception result and execute a wrong move.

**Recommended safeguards at the ROS2 node level:**
- Add a perception confidence gate: if the vision model is below a threshold
  on a letter identity, mark that cell as unknown rather than passing a
  potentially wrong label into the env
- Log every `get_task_sequence` output before execution so failures can be
  diagnosed against the perception input

---

## Decision Gate Summary

```
Phase 1 complete?
  └─ ROS2 integration works in sim ──────────────────► start Phase 2 training
                                                        (in parallel if possible)

Phase 2 complete?
  └─ Loose model trains to C3 beat rate ≥ masked ────► Phase 3 sim validation
  └─ Loose model underperforms ──────────────────────► more training; stay on Phase 1

Phase 3 complete?
  └─ Loose model validated in sim ───────────────────► Phase 4 real robot
  └─ Regression found ───────────────────────────────► diagnose, retrain, retry Phase 3
```

---

## File Reference

| File | Role |
|---|---|
| `models/wordle_ppo_latest.zip` | Phase 1 deployment model — do not overwrite |
| `models/wordle_ppo_loose_latest.zip` | Phase 2/3/4 model — trained separately |
| `training_env/wordle_env.py` | Add `MASK_MODE` support to `action_masks()` |
| `train.py` | Add `MASK_MODE = "tight" \| "loose"` flag |
| `test.py` | `TaskSequencerEvaluator` — inherited by ROS2 node |
| `dictionary.txt` | Full 14,854 five-letter word training set |
| `HOLDOUT_WORD` | `GREAT` — excluded from training, used as generalisation test |
