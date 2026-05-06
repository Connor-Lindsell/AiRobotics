"""
Complex Wordle pick-and-place environment.

Mirrors the physical game board layout:

Board  : x ∈ [-10, 10],  y ∈ [0, 10]
Robot  : home at (0, 0), circle of radius 1
Slots  : 5 × 1 m squares centred at (-2,3) … (2,3)

Letter staging area — a U-shape around the wordle slots with ~3 unit gap:
  • Bottom row  (in front, toward robot):  y = 0,  x = -4.5, -2.25, 0, 2.25, 4.5
  • Left column (left of wordle slots):    x = -5, y = 2, 3, 4
  • Right column (right of wordle slots):  x =  5, y = 2, 3, 4
  Total: 11 staging positions; n_objects tiles are placed at random subset each episode.

Gap reference (centre-to-centre from nearest wordle slot):
  • Bottom row  : gap = 3  (wordle y=3 → staging y=0)
  • Left column : gap = 3  (leftmost wordle x=-2 → staging x=-5)
  • Right column: gap = 3  (rightmost wordle x=2  → staging x=5)
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ============================================================
# Constants  (same workspace / slot layout as simple_env)
# ============================================================

WORD_LENGTH = 5
MAX_OBJECTS = 5
OBS_DIM     = MAX_OBJECTS * 3 + WORD_LENGTH * 2   # 25

WORKSPACE_X_MIN = -10.0
WORKSPACE_X_MAX =  10.0
WORKSPACE_Y_MIN =   0.0
WORKSPACE_Y_MAX =  10.0

ROBOT_HOME   = (0.0, 0.0)
ROBOT_RADIUS = 1.0

SLOT_POSITIONS = [
    (-2.0, 3.0),
    (-1.0, 3.0),
    ( 0.0, 3.0),
    ( 1.0, 3.0),
    ( 2.0, 3.0),
]

# U-shaped letter staging positions (11 total)
U_SHAPE_POSITIONS = [
    # Bottom row — in front of wordle, toward robot  (gap = 3 from wordle centre y=3)
    (-4.5, 5.5), (-3.0, 5.5), (-1.5, 5.5), (0.0, 5.5), (1.5, 5.5), (3.0, 5.5), (4.5, 5.5),
    # Left column — left side of wordle  (gap = 3 from leftmost slot centre x=-2)
    (-5.0, 1.75), (-5.0, 3.0), (-5.0, 4.25),
    # Right column — right side of wordle  (gap = 3 from rightmost slot centre x=2)
    ( 5.0, 1.75), ( 5.0, 3.0), ( 5.0, 4.25),
]

CURRICULUM_STAGES = [
    {"n_objects": 1, "pose_noise_std": 0.0},
    {"n_objects": 2, "pose_noise_std": 0.0},
    {"n_objects": 5, "pose_noise_std": 0.0},
    {"n_objects": 5, "pose_noise_std": 0.02},
]

WORD_LIST = [
    "CRANE", "BRAKE", "STARE", "FLAME", "BRING",
    "CLOCK", "DANCE", "EAGLE", "STONE", "LIGHT",
]

# ============================================================
# Environment
# ============================================================

class ComplexWordleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage: int,
        reward_callback,
        observation_callback,
        target_word=None,
        pose_noise_std=None,
    ):
        super().__init__()
        cfg = CURRICULUM_STAGES[stage - 1]
        self.n_objects            = cfg["n_objects"]
        self.pose_noise_std       = pose_noise_std if pose_noise_std is not None else cfg["pose_noise_std"]
        self._target_word_fixed   = target_word
        self.reward_callback      = reward_callback
        self.observation_callback = observation_callback

        self.action_space      = spaces.Discrete(MAX_OBJECTS * WORD_LENGTH)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.object_poses   = None
        self.object_letters = None
        self.slot_occupied  = None
        self.placed_letters = None
        self.target_word    = None
        self.action_history = []
        self._step_count    = 0

    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if self._target_word_fixed is not None:
            self.target_word = self._target_word_fixed
        elif options.get("target_word"):
            self.target_word = options["target_word"]
        else:
            self.target_word = random.choice(WORD_LIST)

        letters = list(self.target_word[: self.n_objects])
        random.shuffle(letters)
        self.object_letters = letters

        if options.get("poses") is not None:
            self.object_poses = np.array(
                options["poses"][: self.n_objects], dtype=np.float32
            )
        else:
            self.object_poses = self._sample_poses()

        if self.pose_noise_std > 0:
            self.object_poses = (
                self.object_poses
                + self.np_random.normal(0, self.pose_noise_std, self.object_poses.shape).astype(np.float32)
            )

        self.slot_occupied  = np.zeros(WORD_LENGTH, dtype=bool)
        self.placed_letters = [None] * WORD_LENGTH
        self.action_history = []
        self._step_count    = 0

        obs = self.observation_callback(
            self.object_poses, self.object_letters,
            self.slot_occupied, self.placed_letters, self.target_word,
        )
        return obs, {}

    def step(self, action: int):
        object_idx = action // WORD_LENGTH
        slot_idx   = action %  WORD_LENGTH

        obj_pose   = tuple(self.object_poses[object_idx])
        slot_pose  = SLOT_POSITIONS[slot_idx]
        obj_letter = self.object_letters[object_idx]
        tgt_letter = self.target_word[slot_idx]

        self.slot_occupied[slot_idx]  = True
        self.placed_letters[slot_idx] = obj_letter
        self.action_history.append((object_idx, slot_idx))
        self._step_count += 1

        is_correct    = obj_letter == tgt_letter
        n_placed      = sum(1 for l in self.placed_letters if l is not None)
        word_complete = is_correct and (n_placed == self.n_objects)
        is_terminal   = (not is_correct) or word_complete

        reward = self.reward_callback(
            object_letter=obj_letter,
            target_letter=tgt_letter,
            object_pose=obj_pose,
            slot_pose=slot_pose,
            step_count=self._step_count,
            is_terminal=is_terminal,
            word_complete=word_complete,
        )

        obs = self.observation_callback(
            self.object_poses, self.object_letters,
            self.slot_occupied, self.placed_letters, self.target_word,
        )
        return obs, reward, is_terminal, False, {
            "correct": is_correct, "word_complete": word_complete,
        }

    def action_masks(self) -> np.ndarray:
        used = {oi for oi, _ in self.action_history}
        masks = np.zeros(MAX_OBJECTS * WORD_LENGTH, dtype=bool)
        for oi in range(self.n_objects):
            if oi in used:
                continue
            for si in range(WORD_LENGTH):
                if not self.slot_occupied[si]:
                    masks[oi * WORD_LENGTH + si] = True
        return masks

    # ----------------------------------------------------------

    def _sample_poses(self) -> np.ndarray:
        """Pick n_objects positions from U_SHAPE_POSITIONS without replacement."""
        rng = self.np_random if self.np_random is not None else np.random.default_rng()
        indices = rng.choice(len(U_SHAPE_POSITIONS), size=self.n_objects, replace=False)
        poses = [U_SHAPE_POSITIONS[int(i)] for i in indices]
        return np.array(poses, dtype=np.float32)
