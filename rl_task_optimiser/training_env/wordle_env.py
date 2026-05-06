"""
Unified Wordle pick-and-place environment for symbolic task sequencing.

One environment class — WordleSequencingEnv — handles all curriculum stages
C1 through C5 with a fixed observation shape and action space, so MaskablePPO
policy weights transfer continuously across stages.

Position system (18 named positions, all fixed):
  Wordle slots  (pos 0–4)  : 5 positions, centred at y=3
  Staging slots (pos 5–17) : 13 U-shaped positions around the workspace

Action encoding:
  action = source_id * N_POS + dest_id   ∈ Discrete(324)

Observation (624 floats):
  [0:2]      robot position (normalised x, y)
  [2:142]    5 Wordle slots × (occupied, one_hot_letter[26], is_correct)
  [142:493]  13 staging slots × (occupied, one_hot_letter[26])
  [493:623]  target word, 5 × one_hot_letter[26]
  [623:624]  stage indicator (stage / 5)
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ============================================================
# Constants
# ============================================================

WORD_LENGTH = 5
N_WORDLE    = 5
N_STAGING   = 13
N_POS       = N_WORDLE + N_STAGING   # 18
N_LETTERS   = 26

ACTION_DIM  = N_POS * N_POS          # 324

# OBS layout sizes
_WORDLE_BLOCK   = N_WORDLE  * (1 + N_LETTERS + 1)   # 5 × 28 = 140
_STAGING_BLOCK  = N_STAGING * (1 + N_LETTERS)        # 13 × 27 = 351
_TARGET_BLOCK   = WORD_LENGTH * N_LETTERS             # 5 × 26 = 130
OBS_DIM = 2 + _WORDLE_BLOCK + _STAGING_BLOCK + _TARGET_BLOCK + 1  # 624

WORKSPACE_X_MIN, WORKSPACE_X_MAX = -10.0, 10.0
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX =   0.0, 10.0
ROBOT_HOME = np.array([0.0, 0.0], dtype=np.float32)

# Wordle slots 0–4, then staging slots 5–17
ALL_POSITIONS: list[tuple[float, float]] = [
    # Wordle slots (indices 0-4)
    (-2.0, 3.0), (-1.0, 3.0), (0.0, 3.0), (1.0, 3.0), (2.0, 3.0),
    # Staging — bottom row (indices 5-11)
    (-4.5, 5.5), (-3.0, 5.5), (-1.5, 5.5), (0.0, 5.5), (1.5, 5.5), (3.0, 5.5), (4.5, 5.5),
    # Staging — left column (indices 12-14)
    (-5.0, 1.75), (-5.0, 3.0), (-5.0, 4.25),
    # Staging — right column (indices 15-17)
    ( 5.0, 1.75), ( 5.0, 3.0), ( 5.0, 4.25),
]
assert len(ALL_POSITIONS) == N_POS

WORD_LIST = [
    "CRANE", "PLANT", "BRICK", "WATER", "STORM",
    "GRAPE", "FLAME", "CHAIR", "TRAIN", "CLOUD",
]

MAX_STEPS_PER_STAGE = {1: 10, 2: 20, 3: 30, 4: 40, 5: 60}

# Backward-compat aliases used by train.py imports
MAX_OBJECTS = 5


# ============================================================
# Module-level helpers
# ============================================================

def sample_target_word(fixed: str | None = None) -> str:
    return fixed if fixed else random.choice(WORD_LIST)


def one_hot_letter(letter: str | None) -> np.ndarray:
    v = np.zeros(N_LETTERS, dtype=np.float32)
    if letter:
        v[ord(letter) - ord('A')] = 1.0
    return v


def sample_wrong_letter(target_word: str) -> str:
    """Return a random uppercase letter that does not appear in target_word."""
    pool = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in target_word]
    return random.choice(pool)


def norm_pos(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.array([
        (x - WORKSPACE_X_MIN) / (WORKSPACE_X_MAX - WORKSPACE_X_MIN),
        (y - WORKSPACE_Y_MIN) / (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN),
    ], dtype=np.float32)


def euclidean(a: tuple[float, float] | np.ndarray,
              b: tuple[float, float] | np.ndarray) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_travel(robot_pos: np.ndarray,
                   source_pos: tuple[float, float],
                   dest_pos: tuple[float, float]) -> float:
    """Total symbolic travel: robot → source → destination."""
    return euclidean(robot_pos, source_pos) + euclidean(source_pos, dest_pos)


def pos_label(pos_id: int) -> str:
    if pos_id < N_WORDLE:
        return f"T{pos_id}"
    return f"U{pos_id - N_WORDLE}"


# ============================================================
# Environment
# ============================================================

class WordleSequencingEnv(gym.Env):
    """
    Symbolic pick-and-place sequencer for the Wordle robotic task.

    The agent selects symbolic moves (source_position → destination_position).
    Low-level motion (MoveIt, joint control) is handled externally; this env
    only tracks the board state and computes symbolic travel cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage: int,
        reward_callback,
        observation_callback=None,
        target_word: str | None = None,
    ):
        super().__init__()
        if stage not in MAX_STEPS_PER_STAGE:
            raise ValueError(f"stage must be 1–5, got {stage}")

        self.stage     = stage
        self.max_steps = MAX_STEPS_PER_STAGE[stage]
        self._target_word_fixed  = target_word
        self.reward_callback     = reward_callback
        self.observation_callback = observation_callback   # kept for API compat; unused

        self.action_space      = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # State — initialised in reset()
        self.target_word:       str              = ""
        self.position_letter:   list[str | None] = [None] * N_POS
        self.position_occupied: np.ndarray       = np.zeros(N_POS, dtype=bool)
        self.wordle_correct:    np.ndarray       = np.zeros(N_WORDLE, dtype=bool)
        self.robot_pos:         np.ndarray       = ROBOT_HOME.copy()
        self.required_slots:    set[int]         = set()
        self.already_rewarded_slots: set[int]    = set()
        self._step_count: int                    = 0
        self.action_log:  list[str]              = []
        self._cumulative_travel: float           = 0.0
        self._invalid_action_count: int          = 0

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self.target_word = sample_target_word(
            options.get("target_word") or self._target_word_fixed
        )

        self.position_letter   = [None] * N_POS
        self.position_occupied = np.zeros(N_POS, dtype=bool)
        self.wordle_correct    = np.zeros(N_WORDLE, dtype=bool)
        self.robot_pos         = ROBOT_HOME.copy()
        self.already_rewarded_slots = set()
        self.required_slots    = set()
        self._step_count       = 0
        self.action_log        = []
        self._cumulative_travel    = 0.0
        self._invalid_action_count = 0

        _reset_dispatch = {
            1: self._reset_c1,
            2: self._reset_c2,
            3: self._reset_c3,
            4: self._reset_c4,
            5: self._reset_c5,
        }
        _reset_dispatch[self.stage]()

        obs = self._build_obs()
        self._log_initial_state()
        return obs, {}

    # --  Stage reset helpers  ---------------------------------

    def _place(self, pos_id: int, letter: str) -> None:
        self.position_letter[pos_id]   = letter
        self.position_occupied[pos_id] = True

    def _reset_c1(self) -> None:
        """One random letter from target word placed in one random staging slot."""
        wordle_idx   = random.randint(0, N_WORDLE - 1)
        letter       = self.target_word[wordle_idx]
        staging_slot = random.choice(range(N_WORDLE, N_POS))
        self._place(staging_slot, letter)
        self.required_slots = {wordle_idx}

    def _reset_c2(self) -> None:
        """Three random letters from target word placed in three random staging slots."""
        indices      = random.sample(range(N_WORDLE), 3)
        staging_slots = random.sample(range(N_WORDLE, N_POS), 3)
        for wordle_idx, staging_slot in zip(indices, staging_slots):
            self._place(staging_slot, self.target_word[wordle_idx])
        self.required_slots = set(indices)

    def _reset_c3(self) -> None:
        """All five letters placed in five random staging slots. Full word required."""
        staging_slots = random.sample(range(N_WORDLE, N_POS), 5)
        for wordle_idx, staging_slot in enumerate(staging_slots):
            self._place(staging_slot, self.target_word[wordle_idx])
        self.required_slots = set(range(N_WORDLE))

    def _reset_c4(self) -> None:
        """
        One Wordle slot blocked by a wrong letter.
        All five correct letters are available in staging.
        Goal: clear the wrong letter, then fill all 5 Wordle slots.
        """
        blocked = random.randint(0, N_WORDLE - 1)
        self._place(blocked, sample_wrong_letter(self.target_word))

        staging_slots = random.sample(range(N_WORDLE, N_POS), 5)
        for i, slot in enumerate(staging_slots):
            self._place(slot, self.target_word[i])

        self.required_slots = set(range(N_WORDLE))

    def _reset_c5(self) -> None:
        """
        3–5 Wordle slots blocked by wrong letters.
        All five correct letters are available in staging (8 staging slots remain free).
        Goal: clear all wrong letters, fill all 5 Wordle slots correctly.
        """
        n_blocked    = random.randint(3, 5)
        blocked_slots = random.sample(range(N_WORDLE), n_blocked)
        for b in blocked_slots:
            self._place(b, sample_wrong_letter(self.target_word))

        staging_slots = random.sample(range(N_WORDLE, N_POS), 5)
        for i, slot in enumerate(staging_slots):
            self._place(slot, self.target_word[i])

        self.required_slots = set(range(N_WORDLE))

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------

    def step(self, action: int):
        source_id = action // N_POS
        dest_id   = action %  N_POS

        source_pos   = ALL_POSITIONS[source_id]
        dest_pos     = ALL_POSITIONS[dest_id]
        moved_letter = self.position_letter[source_id]

        dist = compute_travel(self.robot_pos, source_pos, dest_pos)
        self._cumulative_travel += dist

        # Classify semantics before mutating state
        src_is_wordle        = source_id < N_WORDLE
        dest_is_wordle       = dest_id   < N_WORDLE
        placing_correct      = dest_is_wordle and (moved_letter == self.target_word[dest_id])
        placing_wrong_wordle = dest_is_wordle and not placing_correct
        clearing_to_staging  = src_is_wordle and not dest_is_wordle
        moving_correct_out   = src_is_wordle and bool(self.wordle_correct[source_id])
        slot_already_rewarded = dest_id in self.already_rewarded_slots

        # Apply move
        self.position_letter[source_id]   = None
        self.position_occupied[source_id] = False
        self.position_letter[dest_id]     = moved_letter
        self.position_occupied[dest_id]   = True
        self.robot_pos = np.array(dest_pos, dtype=np.float32)
        self._step_count += 1

        self._update_wordle_correct()

        word_complete = self._check_word_complete()
        terminated    = word_complete or (self._step_count >= self.max_steps)

        # Human-readable log entry
        src_lbl = pos_label(source_id)
        dst_lbl = pos_label(dest_id)
        self.action_log.append(
            f"Step {self._step_count}: move {moved_letter} from {src_lbl} to {dst_lbl}"
        )

        reward = self.reward_callback(
            source_is_wordle     = src_is_wordle,
            dest_is_wordle       = dest_is_wordle,
            source_is_staging    = not src_is_wordle,
            dest_is_staging      = not dest_is_wordle,
            placing_correct      = placing_correct,
            placing_wrong_wordle = placing_wrong_wordle,
            clearing_to_staging  = clearing_to_staging,
            moving_correct_out   = moving_correct_out,
            word_complete        = word_complete,
            travel_distance      = dist,
            step_count           = self._step_count,
            slot_already_rewarded = slot_already_rewarded,
        )

        if placing_correct and not slot_already_rewarded:
            self.already_rewarded_slots.add(dest_id)

        obs  = self._build_obs()
        info = self._build_info(dist, word_complete, terminated)
        return obs, reward, terminated, False, info

    # ----------------------------------------------------------
    # Action masking (MaskablePPO)
    # ----------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Physical validity mask — blocks impossible moves.
        Semantic wrong placements (wrong letter → wrong Wordle slot) are
        allowed by masking and penalised by the reward signal instead,
        as this gives the policy learning visibility into its mistakes.
        """
        masks = np.zeros(ACTION_DIM, dtype=bool)
        for src in range(N_POS):
            if not self.position_occupied[src]:
                continue
            # Do not allow moving a letter that is already correctly placed
            if src < N_WORDLE and self.wordle_correct[src]:
                continue
            for dst in range(N_POS):
                if dst == src:
                    continue
                if self.position_occupied[dst]:
                    continue
                masks[src * N_POS + dst] = True
        return masks

    # ----------------------------------------------------------
    # Observation
    # ----------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        obs  = np.zeros(OBS_DIM, dtype=np.float32)
        base = 0

        # Robot position (normalised)
        obs[0:2] = norm_pos(tuple(self.robot_pos))
        base = 2

        # Wordle slots: (occupied, one_hot[26], is_correct) × 5
        for i in range(N_WORDLE):
            obs[base]          = float(self.position_occupied[i])
            obs[base+1:base+27] = one_hot_letter(self.position_letter[i])
            obs[base+27]       = float(self.wordle_correct[i])
            base += 28

        # Staging slots: (occupied, one_hot[26]) × 13
        for i in range(N_WORDLE, N_POS):
            obs[base]          = float(self.position_occupied[i])
            obs[base+1:base+27] = one_hot_letter(self.position_letter[i])
            base += 27

        # Target word: one_hot[26] × 5
        for ch in self.target_word:
            obs[base:base+26] = one_hot_letter(ch)
            base += 26

        # Stage indicator
        obs[base] = self.stage / 5.0

        return obs

    # ----------------------------------------------------------
    # Internal state helpers
    # ----------------------------------------------------------

    def _update_wordle_correct(self) -> None:
        for i in range(N_WORDLE):
            self.wordle_correct[i] = bool(
                self.position_occupied[i]
                and self.position_letter[i] == self.target_word[i]
            )

    def _check_word_complete(self) -> bool:
        return all(self.wordle_correct[i] for i in self.required_slots)

    def _build_info(self, dist: float, word_complete: bool, terminated: bool) -> dict:
        board_str = " | ".join(
            f"T{i}={self.position_letter[i] or '?'}"
            + ("✓" if self.wordle_correct[i] else "")
            for i in range(N_WORDLE)
        )
        return {
            "curriculum_stage":   self.stage,
            "target_word":        self.target_word,
            "word_complete":      word_complete,
            "terminated":         terminated,
            "step_count":         self._step_count,
            "n_correct":          int(np.sum(self.wordle_correct)),
            "n_required":         len(self.required_slots),
            "action_log":         list(self.action_log),
            "robot_pos":          tuple(float(v) for v in self.robot_pos),
            "travel_this_step":   dist,
            "cumulative_travel":  self._cumulative_travel,
            "invalid_actions":    self._invalid_action_count,
            "board":              board_str,
        }

    # ----------------------------------------------------------
    # Debug logging
    # ----------------------------------------------------------

    def _log_initial_state(self) -> None:
        staging_occupied = [
            f"{self.position_letter[i]}@U{i - N_WORDLE}"
            for i in range(N_WORDLE, N_POS)
            if self.position_occupied[i]
        ]
        wordle_state = [
            self.position_letter[i] or "?"
            for i in range(N_WORDLE)
        ]
        print(
            f"\n[C{self.stage} Reset] target={self.target_word}"
            f"  required={sorted(self.required_slots)}"
            f"\n  Wordle : {' '.join(wordle_state)}"
            f"\n  Staging: {', '.join(staging_occupied) or 'empty'}"
        )

    def render(self, mode="human"):
        print(f"[C{self.stage}] {self.target_word} | step={self._step_count}")
        for i in range(N_WORDLE):
            ltr = self.position_letter[i] or "_"
            ok  = "✓" if self.wordle_correct[i] else " "
            print(f"  T{i}: {ltr} {ok}  (target={self.target_word[i]})")
        print(f"  robot_pos={tuple(self.robot_pos)}")


# ============================================================
# Factory alias  (train.py does: from training_env.wordle_env import WordleEnv)
# ============================================================

WordleEnv = WordleSequencingEnv
