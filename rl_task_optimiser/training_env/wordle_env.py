"""
Grid-based Wordle pick-and-place environment for symbolic task sequencing.

Board: 13×7 grid of 0.75×0.75 m cells
  x: -4.5 to 4.5  (13 columns, col 0 = x=-4.5)
  y:  0.0 to 4.5  (7 rows,    row 0 = y=0.0)

Robot origin: world (0.0, 0.0) = grid col 6, row 0 = cell_id 6

Wordle zone: 5 cells at row 3 (y=2.25), cols 4–8 (x ∈ {-1.5, -0.75, 0, 0.75, 1.5})
  Cell IDs: [43, 44, 45, 46, 47]

Forbidden staging zone (C4/C5 — non-wordle cells blocked as move destinations):
  Cols 3–9 (x ∈ [-2.25, 2.25])  ×  Rows 0–4 (y ∈ [0, 3.0])
  minus the 5 Wordle cells → 30 blocked destination cells

Action encoding:
  action = source_cell_id * N_CELLS + dest_cell_id  ∈ Discrete(8281)

Observation (2686 floats):
  [0:2]       robot position (normalised x, y)
  [2:2550]    91 cells × (occupied, one_hot_letter[26], is_correct) = 91×28
  [2550:2555] needs_clearing: 5 Wordle slots × 1 (occupied by wrong letter)
  [2555:2685] target word 5 × one_hot_letter[26] = 130
  [2685]      stage indicator (stage / 5)
"""

import math
import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ============================================================
# Grid constants
# ============================================================

GRID_CELL = 0.75
GRID_COLS = 13     # x: -4.5 + col*0.75, col ∈ [0, 12]
GRID_ROWS = 7      # y: row*0.75,         row ∈ [0, 6]
N_CELLS   = GRID_COLS * GRID_ROWS  # 91

WORKSPACE_X_MIN, WORKSPACE_X_MAX = -4.5, 4.5
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX =  0.0, 4.5

ROBOT_HOME = np.array([0.0, 0.0], dtype=np.float32)  # col=6, row=0, cell_id=6

WORD_LENGTH = 5
N_WORDLE    = 5
N_LETTERS   = 26

# Wordle slots: row 3 (y=2.25), cols 4–8 (x ∈ {-1.5, -0.75, 0, 0.75, 1.5})
WORDLE_ROW  = 3
WORDLE_COLS = [4, 5, 6, 7, 8]
WORDLE_CELL_IDS: list[int]       = [WORDLE_ROW * GRID_COLS + c for c in WORDLE_COLS]
WORDLE_CELL_IDS_SET              = frozenset(WORDLE_CELL_IDS)
WORDLE_CELL_ID_TO_IDX: dict[int, int] = {cid: i for i, cid in enumerate(WORDLE_CELL_IDS)}

# Forbidden staging zone for C4/C5: cols 3–9 × rows 0–4, excluding wordle cells
_FORBIDDEN_COLS = frozenset(range(3, 10))  # x ∈ [-2.25, 2.25]
_FORBIDDEN_ROWS = frozenset(range(0, 5))   # y ∈ [0, 3.0]

FORBIDDEN_STAGING_IDS = frozenset(
    i for i in range(N_CELLS)
    if (i % GRID_COLS in _FORBIDDEN_COLS)
    and (i // GRID_COLS in _FORBIDDEN_ROWS)
    and i not in WORDLE_CELL_IDS_SET
)

# All 91 grid positions as (x, y) world coordinates, indexed by cell_id
ALL_POSITIONS: list[tuple[float, float]] = [
    (WORKSPACE_X_MIN + (i % GRID_COLS) * GRID_CELL, (i // GRID_COLS) * GRID_CELL)
    for i in range(N_CELLS)
]

NON_WORDLE_CELLS    = [i for i in range(N_CELLS) if i not in WORDLE_CELL_IDS_SET]
OUTER_STAGING_CELLS = [i for i in NON_WORDLE_CELLS if i not in FORBIDDEN_STAGING_IDS]

ACTION_DIM = N_CELLS * N_CELLS   # 8281

INVALID_ACTION_PENALTY = -50.0

_CELL_BLOCK            = N_CELLS * (1 + N_LETTERS + 1)  # 91 × 28 = 2548
_TARGET_BLOCK          = WORD_LENGTH * N_LETTERS          # 5  × 26 = 130
_NEEDS_CLEARING_BLOCK  = N_WORDLE                         # 5  × 1  = 5
OBS_DIM = 2 + _CELL_BLOCK + _TARGET_BLOCK + _NEEDS_CLEARING_BLOCK + 1  # 2686

MAX_STEPS_PER_STAGE = {1: 10, 2: 15, 3: 25, 4: 35}

# Backward-compat aliases
MAX_OBJECTS = 5
N_POS       = N_CELLS  # used by test.py

def _load_dictionary() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "..", "dictionary.txt")
    fallback = ["CRANE", "PLANT", "BRICK", "WATER", "STORM",
                "GRAPE", "FLAME", "CHAIR", "TRAIN", "CLOUD"]
    try:
        with open(path, encoding="utf-8") as f:
            words = [w.strip().upper() for w in f if len(w.strip()) == 5 and w.strip().isalpha()]
        return words if words else fallback
    except FileNotFoundError:
        return fallback

HOLDOUT_WORD = "GREAT"
_FULL_WORD_LIST = _load_dictionary()
WORD_LIST = [w for w in _FULL_WORD_LIST if w != HOLDOUT_WORD]


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
    return euclidean(robot_pos, source_pos) + euclidean(source_pos, dest_pos)


def pos_label(cell_id: int) -> str:
    if cell_id in WORDLE_CELL_IDS_SET:
        return f"W{WORDLE_CELL_ID_TO_IDX[cell_id]}"
    return f"G{cell_id}"


# ============================================================
# Environment
# ============================================================

class WordleSequencingEnv(gym.Env):
    """
    Grid-based symbolic pick-and-place sequencer for the Wordle robotic task.

    The agent selects symbolic moves (source_cell → dest_cell) on a 13×7 grid.
    Low-level motion is handled externally; this env tracks board state and
    computes symbolic travel cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage: int,
        reward_callback,
        observation_callback=None,   # kept for API compat; unused
        target_word: str | None = None,
        fixed_initial_positions: dict[int, str] | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        if stage not in MAX_STEPS_PER_STAGE:
            raise ValueError(f"stage must be 1–4, got {stage}")

        self.stage     = stage
        self.max_steps = MAX_STEPS_PER_STAGE[stage]
        self._target_word_fixed      = target_word
        self._fixed_initial_positions = fixed_initial_positions
        self.reward_callback         = reward_callback
        self.observation_callback    = observation_callback
        self.verbose = verbose

        self.action_space      = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # State — initialised in reset()
        self.target_word:            str              = ""
        self.position_letter:        list[str | None] = [None] * N_CELLS
        self.position_occupied:      np.ndarray       = np.zeros(N_CELLS, dtype=bool)
        self.wordle_correct:         np.ndarray       = np.zeros(N_WORDLE, dtype=bool)
        self.robot_pos:              np.ndarray       = ROBOT_HOME.copy()
        self.required_slots:         set[int]         = set()   # wordle cell IDs
        self.already_rewarded_slots: set[int]         = set()   # wordle cell IDs
        self._step_count:            int              = 0
        self.action_log:             list[str]        = []
        self._cumulative_travel:     float            = 0.0
        self._invalid_action_count:  int              = 0

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self.target_word = sample_target_word(
            options.get("target_word") or self._target_word_fixed
        )

        self.position_letter        = [None] * N_CELLS
        self.position_occupied      = np.zeros(N_CELLS, dtype=bool)
        self.wordle_correct         = np.zeros(N_WORDLE, dtype=bool)
        self.robot_pos              = ROBOT_HOME.copy()
        self.already_rewarded_slots = set()
        self.required_slots         = set()
        self._step_count            = 0
        self.action_log             = []
        self._cumulative_travel     = 0.0
        self._invalid_action_count  = 0

        if self._fixed_initial_positions is not None:
            for cell_id, letter in self._fixed_initial_positions.items():
                self._place(cell_id, letter)
            self.required_slots = set(WORDLE_CELL_IDS)
        else:
            _reset_dispatch = {
                1: self._reset_c1,
                2: self._reset_c2,
                3: self._reset_c3,
                4: self._reset_c4,
            }
            _reset_dispatch[self.stage]()

        obs = self._build_obs()
        if self.verbose:
            self._log_initial_state()
        return obs, {}

    # -- Stage reset helpers  ----------------------------------

    def _place(self, cell_id: int, letter: str) -> None:
        self.position_letter[cell_id]   = letter
        self.position_occupied[cell_id] = True

    def _reset_c1(self) -> None:
        # 5 correct letters in random staging, no distractors
        cells = random.sample(NON_WORDLE_CELLS, 5)
        for wi, cell in enumerate(cells):
            self._place(cell, self.target_word[wi])
        self.required_slots = set(WORDLE_CELL_IDS)

    def _reset_c2(self) -> None:
        # 5 correct letters + 5 distractors in random staging
        cells = random.sample(NON_WORDLE_CELLS, 10)
        for wi, cell in enumerate(cells[:5]):
            self._place(cell, self.target_word[wi])
        for cell in cells[5:]:
            self._place(cell, sample_wrong_letter(self.target_word))
        self.required_slots = set(WORDLE_CELL_IDS)

    def _reset_c3(self) -> None:
        # All 5 Wordle slots blocked; 5 correct + 10 distractors in outer staging
        for wi in range(N_WORDLE):
            self._place(WORDLE_CELL_IDS[wi], sample_wrong_letter(self.target_word))
        cells = random.sample(OUTER_STAGING_CELLS, 15)
        for wi, cell in enumerate(cells[:5]):
            self._place(cell, self.target_word[wi])
        for cell in cells[5:]:
            self._place(cell, sample_wrong_letter(self.target_word))
        self.required_slots = set(WORDLE_CELL_IDS)

    def _reset_c4(self) -> None:
        # Identical board to C3 — difficulty comes from the loose action mask
        for wi in range(N_WORDLE):
            self._place(WORDLE_CELL_IDS[wi], sample_wrong_letter(self.target_word))
        cells = random.sample(OUTER_STAGING_CELLS, 15)
        for wi, cell in enumerate(cells[:5]):
            self._place(cell, self.target_word[wi])
        for cell in cells[5:]:
            self._place(cell, sample_wrong_letter(self.target_word))
        self.required_slots = set(WORDLE_CELL_IDS)

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------

    def step(self, action: int):
        source_id = action // N_CELLS
        dest_id   = action %  N_CELLS

        robot_pos_before = self.robot_pos.copy()
        source_pos       = ALL_POSITIONS[source_id]
        dest_pos         = ALL_POSITIONS[dest_id]

        # Reject any action the mask would block
        if not self.action_masks()[action]:
            self._step_count          += 1
            self._invalid_action_count += 1
            terminated = self._step_count >= self.max_steps
            obs  = self._build_obs()
            info = self._build_info(0.0, False, terminated,
                                    robot_pos_before, source_pos, dest_pos, [])
            return obs, INVALID_ACTION_PENALTY, terminated, False, info

        moved_letter = self.position_letter[source_id]

        dist = compute_travel(self.robot_pos, source_pos, dest_pos)
        self._cumulative_travel += dist

        src_is_wordle  = source_id in WORDLE_CELL_IDS_SET
        dest_is_wordle = dest_id   in WORDLE_CELL_IDS_SET

        src_wi  = WORDLE_CELL_ID_TO_IDX.get(source_id)
        dest_wi = WORDLE_CELL_ID_TO_IDX.get(dest_id)

        placing_correct      = dest_is_wordle and (moved_letter == self.target_word[dest_wi])
        placing_wrong_wordle = dest_is_wordle and not placing_correct
        clearing_to_staging  = src_is_wordle  and not dest_is_wordle
        moving_correct_out   = src_is_wordle  and bool(self.wordle_correct[src_wi])
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

        self.action_log.append(
            f"Step {self._step_count}: move {moved_letter} "
            f"from {pos_label(source_id)} to {pos_label(dest_id)}"
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

        path_segments = [
            (tuple(robot_pos_before), source_pos),
            (source_pos, dest_pos),
        ]
        obs  = self._build_obs()
        info = self._build_info(dist, word_complete, terminated,
                                robot_pos_before, source_pos, dest_pos, path_segments)
        return obs, reward, terminated, False, info

    # ----------------------------------------------------------
    # Action masking (MaskablePPO)
    # ----------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        masks = np.zeros(ACTION_DIM, dtype=bool)

        if self.stage <= 2:
            # C1-C2: tight — only correct letter into correct empty required slot
            for src in range(N_CELLS):
                if not self.position_occupied[src]:
                    continue
                letter = self.position_letter[src]
                for dst in self.required_slots:
                    if dst == src or self.position_occupied[dst]:
                        continue
                    if letter == self.target_word[WORDLE_CELL_ID_TO_IDX[dst]]:
                        masks[src * N_CELLS + dst] = True

        elif self.stage == 3:
            # C3: semi-constrained
            #   From Wordle: free clearing to any non-forbidden empty staging cell
            #   From staging: only correct letter → correct empty Wordle slot
            for src in range(N_CELLS):
                if not self.position_occupied[src]:
                    continue
                if src in WORDLE_CELL_IDS_SET:
                    wi = WORDLE_CELL_ID_TO_IDX[src]
                    if self.wordle_correct[wi]:
                        continue  # never evict a correctly placed letter
                    for dst in NON_WORDLE_CELLS:
                        if self.position_occupied[dst]:
                            continue
                        if dst in FORBIDDEN_STAGING_IDS:
                            continue
                        masks[src * N_CELLS + dst] = True
                else:
                    letter = self.position_letter[src]
                    for dst in self.required_slots:
                        if dst == src or self.position_occupied[dst]:
                            continue
                        if letter == self.target_word[WORDLE_CELL_ID_TO_IDX[dst]]:
                            masks[src * N_CELLS + dst] = True

        else:
            # C4: loose — any move to empty non-forbidden cell, never evict correct letters
            for src in range(N_CELLS):
                if not self.position_occupied[src]:
                    continue
                if src in WORDLE_CELL_IDS_SET:
                    wi = WORDLE_CELL_ID_TO_IDX[src]
                    if self.wordle_correct[wi]:
                        continue
                for dst in range(N_CELLS):
                    if dst == src or self.position_occupied[dst]:
                        continue
                    if dst in FORBIDDEN_STAGING_IDS:
                        continue
                    masks[src * N_CELLS + dst] = True

        return masks

    # ----------------------------------------------------------
    # Observation
    # ----------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        obs  = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0:2] = norm_pos(tuple(self.robot_pos))
        base = 2

        for cell_id in range(N_CELLS):
            obs[base]           = float(self.position_occupied[cell_id])
            obs[base+1:base+27] = one_hot_letter(self.position_letter[cell_id])
            if cell_id in WORDLE_CELL_IDS_SET:
                wi = WORDLE_CELL_ID_TO_IDX[cell_id]
                obs[base+27] = float(self.wordle_correct[wi])
            base += 28

        for wi, cid in enumerate(WORDLE_CELL_IDS):
            obs[base + wi] = float(
                self.position_occupied[cid] and not self.wordle_correct[wi]
            )
        base += N_WORDLE

        for ch in self.target_word:
            obs[base:base+26] = one_hot_letter(ch)
            base += 26

        obs[base] = self.stage / 5.0
        return obs

    # ----------------------------------------------------------
    # Internal state helpers
    # ----------------------------------------------------------

    def _update_wordle_correct(self) -> None:
        for wi, cid in enumerate(WORDLE_CELL_IDS):
            self.wordle_correct[wi] = bool(
                self.position_occupied[cid]
                and self.position_letter[cid] == self.target_word[wi]
            )

    def _check_word_complete(self) -> bool:
        for cid in self.required_slots:
            wi = WORDLE_CELL_ID_TO_IDX[cid]
            if not self.wordle_correct[wi]:
                return False
        return True

    def _build_info(self, dist: float, word_complete: bool, terminated: bool,
                    robot_pos_before: np.ndarray,
                    source_pos: tuple[float, float],
                    dest_pos: tuple[float, float],
                    path_segments: list) -> dict:
        board_str = " | ".join(
            f"W{wi}={self.position_letter[cid] or '?'}"
            + ("✓" if self.wordle_correct[wi] else "")
            for wi, cid in enumerate(WORDLE_CELL_IDS)
        )
        return {
            "curriculum_stage":  self.stage,
            "target_word":       self.target_word,
            "word_complete":     word_complete,
            "terminated":        terminated,
            "step_count":        self._step_count,
            "n_correct":         int(np.sum(self.wordle_correct)),
            "n_required":        len(self.required_slots),
            "action_log":        list(self.action_log),
            "robot_pos":         tuple(float(v) for v in self.robot_pos),
            "travel_this_step":  dist,
            "cumulative_travel": self._cumulative_travel,
            "invalid_actions":   self._invalid_action_count,
            "board":             board_str,
            "robot_pos_before":  tuple(float(v) for v in robot_pos_before),
            "source_pos":        source_pos,
            "dest_pos":          dest_pos,
            "path_segments":     path_segments,
        }

    # ----------------------------------------------------------
    # Debug logging
    # ----------------------------------------------------------

    def _log_initial_state(self) -> None:
        staging_occupied = [
            f"{self.position_letter[i]}@G{i}"
            for i in range(N_CELLS)
            if i not in WORDLE_CELL_IDS_SET and self.position_occupied[i]
        ]
        wordle_state = [
            self.position_letter[cid] or "?"
            for cid in WORDLE_CELL_IDS
        ]
        print(
            f"\n[C{self.stage} Reset] target={self.target_word}"
            f"  required={sorted(self.required_slots)}"
            f"\n  Wordle : {' '.join(wordle_state)}"
            f"\n  Staging: {', '.join(staging_occupied) or 'empty'}"
        )

    def render(self, mode="human"):
        print(f"[C{self.stage}] {self.target_word} | step={self._step_count}")
        for wi, cid in enumerate(WORDLE_CELL_IDS):
            ltr = self.position_letter[cid] or "_"
            ok  = "✓" if self.wordle_correct[wi] else " "
            print(f"  W{wi}: {ltr} {ok}  (target={self.target_word[wi]})")
        print(f"  robot_pos={tuple(self.robot_pos)}")


# ============================================================
# Factory alias
# ============================================================

WordleEnv = WordleSequencingEnv
