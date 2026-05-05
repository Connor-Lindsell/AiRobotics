import os
import math
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ============================================================
# Hyperparameters
# ============================================================
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE   = 3e-4
N_STEPS         = 2048
BATCH_SIZE      = 64
SAVE_FREQ       = 10_000

# ============================================================
# Reward Function Configuration Parameters  (mirrors quiz2 pattern)
# ============================================================
# Motion cost multiplier — applied to estimated Euclidean distance (metres).
# Negative value means moving further costs more reward.
# TODO: Tune once real workspace geometry is confirmed.
TIME_COST_SCALE          = -1.0

# Terminal reward for placing a letter in the CORRECT slot.
CORRECT_PLACEMENT_BONUS  =  50.0

# Terminal penalty for placing a letter in the WRONG slot (ends episode).
WRONG_PLACEMENT_PENALTY  = -100.0

# Bonus added when ALL slots are filled correctly (word complete).
WORD_COMPLETE_BONUS      =  200.0

# ============================================================
# Curriculum control
# ============================================================
# Set to the stage currently being trained (1–4).
# Advance manually between runs — bump this constant and re-run train.py.
# Each run resumes from the _latest checkpoint via the resume logic below.
# Stage definitions live in env/wordle_env.py: CURRICULUM_STAGES.
# TODO: Optionally implement an auto-advance callback that monitors
#   mean episode reward and advances the stage when it crosses a threshold.
CURRICULUM_STAGE = 1

# ============================================================
# Arm home position (robot base frame, metres)
# ============================================================
# Used by custom_reward() to estimate motion cost from home to first object.
# TODO: Replace with the actual UR3e home/ready joint config projected into
#   the workspace plane (z=0 slice of the TCP position).
ARM_HOME_POS = (0.0, 0.0)

# ============================================================
# Versioned saving and logging  (mirrors quiz2 pattern)
# ============================================================
MODEL_DIR  = "models"
MODEL_NAME = "wordle_ppo"
LOGS_DIR   = "logs"
LOG_FILE   = "training_log.txt"

# MaskablePPO action-masking works cleanly with a single env or DummyVecEnv.
# SubprocVecEnv serialises the env into a subprocess, making action_masks()
# harder to call. Set N_ENVS=1 for now; investigate DummyVecEnv if parallelism needed.
N_ENVS = 1


# ============================================================
# Observation callback
# ============================================================
def custom_observation(object_poses, object_letters, slot_occupied, placed_letters, target_word):
    """
    Build the flat float32 observation vector for the neural network.

    Injected into WordleEnv as observation_callback — mirrors quiz2's
    observation_callback injection pattern.

    Args:
        object_poses    (np.ndarray): shape (n_objects, 2) — (x, y) per remaining object
        object_letters  (list[str]):  letter identity per object, length n_objects
        slot_occupied   (np.ndarray): bool array, shape (WORD_LENGTH,)
        placed_letters  (list[str|None]): placed letter per slot, None if empty
        target_word     (str):        5-letter target word

    Returns:
        np.ndarray: flat float32 observation vector of length OBS_DIM (25)

    TODO: Import WORKSPACE_X_MIN/MAX, WORKSPACE_Y_MIN/MAX, MAX_OBJECTS, WORD_LENGTH, OBS_DIM
      from env.wordle_env for normalisation bounds and sizing.

    TODO: Build flat vector with this layout (matches WordleEnv._compute_obs() fallback):
        Positions 0–14: [x_norm_i, y_norm_i, letter_enc_i] for i in 0..MAX_OBJECTS-1
            x_norm = (x - WORKSPACE_X_MIN) / (WORKSPACE_X_MAX - WORKSPACE_X_MIN)
            y_norm = (y - WORKSPACE_Y_MIN) / (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN)
            letter_enc = (ord(letter) - ord('A')) / 25.0
            Absent objects (index >= n_objects) left as 0.0 (already zeroed)
        Positions 15–19: slot_occupied as float32 (0.0 = empty, 1.0 = filled)
        Positions 20–24: (ord(target_word[i]) - ord('A')) / 25.0 per slot
    """
    # TODO: from env.wordle_env import OBS_DIM, MAX_OBJECTS, WORD_LENGTH
    # TODO:   WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX
    # TODO: obs = np.zeros(OBS_DIM, dtype=np.float32)
    # TODO: for i, (pose, letter) in enumerate(zip(object_poses, object_letters)):
    #           x_norm = (pose[0] - WORKSPACE_X_MIN) / (WORKSPACE_X_MAX - WORKSPACE_X_MIN)
    #           y_norm = (pose[1] - WORKSPACE_Y_MIN) / (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN)
    #           l_enc  = (ord(letter) - ord('A')) / 25.0
    #           obs[i*3 : i*3+3] = [x_norm, y_norm, l_enc]
    # TODO: obs[MAX_OBJECTS*3 : MAX_OBJECTS*3 + WORD_LENGTH] = slot_occupied.astype(np.float32)
    # TODO: for j, letter in enumerate(target_word):
    #           obs[MAX_OBJECTS*3 + WORD_LENGTH + j] = (ord(letter) - ord('A')) / 25.0
    # TODO: return obs
    raise NotImplementedError("custom_observation not yet implemented — see TODO above")


# ============================================================
# Reward callback
# ============================================================
def custom_reward(
    object_letter,
    target_letter,
    object_pose,
    slot_pose,
    step_count,
    is_terminal,
    word_complete,
):
    """
    Compute the scalar reward for a single pick-and-place action.

    Injected into WordleEnv as reward_callback — mirrors quiz2's
    reward_callback injection pattern.

    Args:
        object_letter (str):          letter of the object just picked
        target_letter (str):          correct letter for the target slot (target_word[slot_idx])
        object_pose   (tuple[float]): (x, y) position of the picked object (metres)
        slot_pose     (tuple[float]): (x, y) position of the target slot (metres, from SLOT_POSITIONS)
        step_count    (int):          current step index in the episode (0-based)
        is_terminal   (bool):         True if this action ends the episode
        word_complete (bool):         True if all WORD_LENGTH slots are now correctly filled

    Returns:
        float: scalar reward

    Reward formula:
        1. Motion cost (every step):
               reward += TIME_COST_SCALE * (dist(ARM_HOME_POS, object_pose) + dist(object_pose, slot_pose))
           For step_count > 0 the arm starts from the previous slot, not ARM_HOME_POS.
           TODO: Track last_slot_pose in the env and pass it through the callback,
                 or adjust the formula to always use ARM_HOME_POS as a simplification.

        2. Correct placement (is_terminal and object_letter == target_letter):
               reward += CORRECT_PLACEMENT_BONUS

        3. Wrong placement (is_terminal and object_letter != target_letter):
               reward += WRONG_PLACEMENT_PENALTY
           Large negative to discourage random/lazy placement.

        4. Word complete (word_complete):
               reward += WORD_COMPLETE_BONUS
           Stacked on top of the final correct-placement bonus.
    """
    reward = 0.0

    # TODO: dist_home_to_obj = math.sqrt((object_pose[0] - ARM_HOME_POS[0])**2 +
    #                                     (object_pose[1] - ARM_HOME_POS[1])**2)
    # TODO: dist_obj_to_slot = math.sqrt((slot_pose[0]   - object_pose[0])**2 +
    #                                     (slot_pose[1]   - object_pose[1])**2)
    # TODO: reward += TIME_COST_SCALE * (dist_home_to_obj + dist_obj_to_slot)

    # TODO: if is_terminal and object_letter == target_letter:
    #           reward += CORRECT_PLACEMENT_BONUS
    # TODO: elif is_terminal and object_letter != target_letter:
    #           reward += WRONG_PLACEMENT_PENALTY
    # TODO: if word_complete:
    #           reward += WORD_COMPLETE_BONUS

    raise NotImplementedError("custom_reward not yet implemented — see TODO above")
    return reward  # noqa: unreachable — remove once TODO is implemented


# ============================================================
# Versioned model saving helpers  (mirrors quiz2 pattern)
# ============================================================
def get_next_version() -> int:
    """Find the next unused version number for model saving."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    v = 1
    while os.path.exists(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{v}.zip")):
        v += 1
    return v


def save_training_log(version, model, total_timesteps, curriculum_stage):
    """
    Append a training run summary to LOG_FILE.

    Mirrors quiz2's save_training_log() exactly — same section structure,
    same ep_info_buffer extraction pattern.
    """
    from datetime import datetime
    buf    = model.ep_info_buffer
    ep_rew = round(float(np.mean([e["r"] for e in buf])), 2) if buf else "N/A"
    ep_len = round(float(np.mean([e["l"] for e in buf])), 2) if buf else "N/A"

    # TODO: Uncomment and implement once training is functional:
    # with open(LOG_FILE, "a") as f:
    #     f.write(f"\n{'='*48}\n")
    #     f.write(f"Run v{version}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Stage {curriculum_stage}\n")
    #     f.write(f"\n  -- Results --\n")
    #     f.write(f"  total_timesteps         : {total_timesteps}\n")
    #     f.write(f"  ep_rew_mean             : {ep_rew}\n")
    #     f.write(f"  ep_len_mean             : {ep_len}\n")
    #     f.write(f"\n  -- Reward Config --\n")
    #     f.write(f"  TIME_COST_SCALE         : {TIME_COST_SCALE}\n")
    #     f.write(f"  CORRECT_PLACEMENT_BONUS : {CORRECT_PLACEMENT_BONUS}\n")
    #     f.write(f"  WRONG_PLACEMENT_PENALTY : {WRONG_PLACEMENT_PENALTY}\n")
    #     f.write(f"  WORD_COMPLETE_BONUS     : {WORD_COMPLETE_BONUS}\n")
    #     f.write(f"  CURRICULUM_STAGE        : {curriculum_stage}\n")
    #     f.write(f"\n  -- PPO Hyperparameters --\n")
    #     f.write(f"  TOTAL_TIMESTEPS         : {total_timesteps}\n")
    #     f.write(f"  N_ENVS                  : {N_ENVS}\n")
    #     f.write(f"  learning_rate           : {model.learning_rate}\n")
    #     f.write(f"  n_steps                 : {model.n_steps}\n")
    #     f.write(f"  batch_size              : {model.batch_size}\n")
    #     f.write(f"  ent_coef                : {model.ent_coef}\n")
    #     f.write(f"{'='*48}\n")
    raise NotImplementedError("save_training_log not yet implemented — uncomment block above")


# ============================================================
# Environment factory
# ============================================================
def make_env():
    """
    Create a WordleEnv instance with callback injection.

    Mirrors quiz2's env_kwargs pattern — reward and observation logic is
    injected as callbacks, keeping the env class and training script decoupled.
    This allows reward shaping to be iterated in train.py without touching the env.
    """
    from env.wordle_env import WordleEnv
    # TODO: return WordleEnv(
    #     stage                = CURRICULUM_STAGE,
    #     reward_callback      = custom_reward,
    #     observation_callback = custom_observation,
    # )
    raise NotImplementedError("make_env() not yet wired to WordleEnv — see TODO above")


# ============================================================
# Training entry point
# ============================================================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)

    env = make_env()

    # --- Resume from latest checkpoint if one exists (curriculum training) ---
    # Mirrors quiz2's resume-from-latest pattern exactly.
    # Workflow for advancing curriculum stages:
    #   1. Bump CURRICULUM_STAGE above.
    #   2. Re-run train.py — the previous stage's _latest save is loaded
    #      and training continues from the same TensorBoard step counter.
    latest_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    if os.path.exists(latest_path + ".zip"):
        print(f"Resuming from {latest_path}.zip  (Stage {CURRICULUM_STAGE}) ...")
        # TODO: model = MaskablePPO.load(latest_path, env=env, tensorboard_log=LOGS_DIR)
        raise NotImplementedError("Resume-from-latest not yet implemented — see TODO above")
    else:
        print(f"No previous model found — training from scratch (Stage {CURRICULUM_STAGE}).")
        # TODO: model = MaskablePPO(
        #     "MlpPolicy",
        #     env,
        #     learning_rate   = LEARNING_RATE,
        #     n_steps         = N_STEPS,
        #     batch_size      = BATCH_SIZE,
        #     ent_coef        = 0.01,
        #     tensorboard_log = LOGS_DIR,
        #     verbose         = 1,
        # )
        raise NotImplementedError("Fresh MaskablePPO construction not yet implemented — see TODO above")

    # --- Checkpoint callback ---
    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
    )

    # --- Train ---
    # reset_num_timesteps=False keeps the TensorBoard step counter continuous
    # across curriculum stage transitions — matches quiz2 behaviour.
    # TODO: model.learn(
    #     total_timesteps     = TOTAL_TIMESTEPS,
    #     callback            = checkpoint_callback,
    #     reset_num_timesteps = False,
    # )

    # --- Versioned save + overwrite _latest (mirrors quiz2) ---
    # TODO: version = get_next_version()
    # TODO: model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{version}"))
    # TODO: model.save(latest_path)
    # TODO: save_training_log(version, model, model.num_timesteps, CURRICULUM_STAGE)
    # TODO: print(f"Saved  ->  {MODEL_DIR}/{MODEL_NAME}_v{{version}}.zip  +  {latest_path}.zip")
    # TODO: print(f"Log    ->  {LOG_FILE}")
