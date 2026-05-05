import os
import math
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ============================================================
# PPO Hyperparameters
# ============================================================
# Total environment steps per training run. Later curriculum stages typically need
# more steps to converge — increase if mean episode reward has not plateaued.
TOTAL_TIMESTEPS = 100_000

# Adam optimizer step size. 3e-4 is a reliable default for MaskablePPO on discrete tasks.
# Lower values slow convergence but reduce the risk of a gradient update destabilising
# an already-good policy.
LEARNING_RATE   = 3e-4

# Steps collected per environment before each gradient update. Larger values produce
# more stable gradient estimates at the cost of slower iteration — the policy is only
# updated once per N_STEPS environment interactions.
N_STEPS         = 2048

# Minibatch size used during the gradient update phase. Must divide N_STEPS evenly.
# Smaller batches add gradient noise (a regularising effect); larger batches give
# smoother updates but can converge to sharper minima.
BATCH_SIZE      = 64

# Save a checkpoint every SAVE_FREQ environment steps so training can be recovered
# if a run diverges or is interrupted.
SAVE_FREQ       = 10_000

# ============================================================
# Reward shaping parameters
# ============================================================
# Per-step motion cost multiplier applied to total Euclidean distance travelled (metres).
# reward += TIME_COST_SCALE * (dist(home → object) + dist(object → slot))
# More negative values push the agent toward shorter paths. Zero disables motion cost.
TIME_COST_SCALE          = -1.0

# Reward for placing a letter in the correct slot. Must be large enough to outweigh
# the expected cumulative motion cost of a successful episode, otherwise the agent
# may prefer ending episodes early via a wrong placement.
CORRECT_PLACEMENT_BONUS  =  50.0

# Penalty applied when a letter is placed in the wrong slot; also terminates the episode.
# Should exceed the maximum possible motion cost across a full episode so that the
# agent never finds a wrong placement preferable to continued correct play.
WRONG_PLACEMENT_PENALTY  = -100.0

# Additional terminal bonus awarded when all five slots are filled correctly.
# Provides a strong completion signal on top of the final per-letter bonus, encouraging
# the agent to finish the whole word rather than stopping after early correct placements.
WORD_COMPLETE_BONUS      =  200.0

# ============================================================
# Curriculum stage
# ============================================================
# Active stage (1–4). Advance manually between runs: bump this constant and re-run
# train.py. The script auto-resumes from the latest checkpoint so learned weights
# transfer across stages. Stage definitions (n_objects, pose_noise_std) live in
# env/wordle_env.py: CURRICULUM_STAGES.
CURRICULUM_STAGE = 1

# ============================================================
# Robot arm home position
# ============================================================
# (x, y) in metres, robot base frame. Used by custom_reward() to compute the motion
# cost component for each pick — the arm is assumed to start here at the beginning
# of every episode and after each placement.
ARM_HOME_POS = (0.0, 0.0)

# ============================================================
# Model and log paths
# ============================================================
MODEL_DIR  = "models"
MODEL_NAME = "wordle_ppo"
LOGS_DIR   = "logs"
LOG_FILE   = "training_log.txt"

# MaskablePPO requires action_masks() to be called on the env during rollout
# collection. A single env works cleanly; SubprocVecEnv serialises the env into a
# subprocess, making action_masks() harder to call.
N_ENVS = 1


# ============================================================
# Observation callback
# ============================================================
def custom_observation(object_poses, object_letters, slot_occupied, placed_letters, target_word):
    """
    Build the flat float32 observation vector fed to the policy network.

    Injected into WordleEnv as observation_callback so reward and observation logic
    can be iterated in train.py without modifying the environment class.

    Args:
        object_poses    (np.ndarray): shape (n_objects, 2) — (x, y) per remaining object, metres
        object_letters  (list[str]):  uppercase letter identity per object, length n_objects
        slot_occupied   (np.ndarray): bool array, shape (WORD_LENGTH,) — True if slot is filled
        placed_letters  (list[str|None]): placed letter per slot, None if slot is empty
        target_word     (str):        5-letter uppercase target word

    Returns:
        np.ndarray: flat float32 vector of length OBS_DIM (25)

    Vector layout (25 floats total):
        [0  – 14] Object block — 3 floats per object slot (5 slots × 3):
                    x_norm  = (x - WORKSPACE_X_MIN) / (WORKSPACE_X_MAX - WORKSPACE_X_MIN)
                    y_norm  = (y - WORKSPACE_Y_MIN) / (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN)
                    l_enc   = (ord(letter) - ord('A')) / 25.0
                  Unused object slots (index >= n_objects) are left as 0.0.
        [15 – 19] Slot occupancy — slot_occupied cast to float32 (0.0 empty, 1.0 filled)
        [20 – 24] Target encoding — (ord(target_word[i]) - ord('A')) / 25.0 per slot
    """
    from env.wordle_env import (
        OBS_DIM, MAX_OBJECTS, WORD_LENGTH,
        WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    )
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for i, (pose, letter) in enumerate(zip(object_poses, object_letters)):
        x_norm = (pose[0] - WORKSPACE_X_MIN) / (WORKSPACE_X_MAX - WORKSPACE_X_MIN)
        y_norm = (pose[1] - WORKSPACE_Y_MIN) / (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN)
        l_enc  = (ord(letter) - ord('A')) / 25.0
        obs[i*3 : i*3+3] = [x_norm, y_norm, l_enc]
    obs[MAX_OBJECTS*3 : MAX_OBJECTS*3 + WORD_LENGTH] = slot_occupied.astype(np.float32)
    for j, letter in enumerate(target_word):
        obs[MAX_OBJECTS*3 + WORD_LENGTH + j] = (ord(letter) - ord('A')) / 25.0
    return obs


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

    Injected into WordleEnv as reward_callback. Adjusting the constants at the top
    of this file (TIME_COST_SCALE, CORRECT_PLACEMENT_BONUS, etc.) reshapes the
    reward signal without touching the environment class.

    Args:
        object_letter (str):          uppercase letter of the object just picked
        target_letter (str):          correct letter for the target slot (target_word[slot_idx])
        object_pose   (tuple[float]): (x, y) position of the picked object, metres
        slot_pose     (tuple[float]): (x, y) position of the target slot, metres
        step_count    (int):          current step index in the episode (0-based)
        is_terminal   (bool):         True if this action ends the episode
        word_complete (bool):         True if all WORD_LENGTH slots are now correctly filled

    Returns:
        float: scalar reward for this step

    Reward components (all applied additively):
        Motion cost (every step):
            TIME_COST_SCALE * (dist(ARM_HOME_POS → object_pose) + dist(object_pose → slot_pose))
        Correct placement (is_terminal and object_letter == target_letter):
            + CORRECT_PLACEMENT_BONUS
        Wrong placement (is_terminal and object_letter != target_letter):
            + WRONG_PLACEMENT_PENALTY
        Word complete (word_complete):
            + WORD_COMPLETE_BONUS  (stacked on top of the final correct-placement bonus)
    """
    reward = 0.0

    dist_home_to_obj = math.sqrt((object_pose[0] - ARM_HOME_POS[0])**2 +
                                  (object_pose[1] - ARM_HOME_POS[1])**2)
    dist_obj_to_slot = math.sqrt((slot_pose[0]   - object_pose[0])**2 +
                                  (slot_pose[1]   - object_pose[1])**2)
    reward += TIME_COST_SCALE * (dist_home_to_obj + dist_obj_to_slot)

    if is_terminal and object_letter == target_letter:
        reward += CORRECT_PLACEMENT_BONUS
    elif is_terminal and object_letter != target_letter:
        reward += WRONG_PLACEMENT_PENALTY
    if word_complete:
        reward += WORD_COMPLETE_BONUS

    return reward


# ============================================================
# Versioned model saving helpers
# ============================================================
def get_next_version() -> int:
    """Return the lowest unused version number, scanning MODEL_DIR for existing saves."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    v = 1
    while os.path.exists(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{v}.zip")):
        v += 1
    return v


def save_training_log(version, model, total_timesteps, curriculum_stage):
    """
    Append a training run summary to LOG_FILE.

    Extracts mean episode reward and length from model.ep_info_buffer (the ring buffer
    of recent completed episodes maintained by SB3 during learn()). If the buffer is
    empty (e.g. no episode completed in this run), fields are logged as "N/A".

    Args:
        version          (int):        version number assigned to this run's saved model
        model            (MaskablePPO): trained model instance
        total_timesteps  (int):        cumulative timesteps at end of this run
        curriculum_stage (int):        CURRICULUM_STAGE value used for this run
    """
    from datetime import datetime
    buf    = model.ep_info_buffer
    ep_rew = round(float(np.mean([e["r"] for e in buf])), 2) if buf else "N/A"
    ep_len = round(float(np.mean([e["l"] for e in buf])), 2) if buf else "N/A"

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*48}\n")
        f.write(f"Run v{version}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Stage {curriculum_stage}\n")
        f.write(f"\n  -- Results --\n")
        f.write(f"  total_timesteps         : {total_timesteps}\n")
        f.write(f"  ep_rew_mean             : {ep_rew}\n")
        f.write(f"  ep_len_mean             : {ep_len}\n")
        f.write(f"\n  -- Reward Config --\n")
        f.write(f"  TIME_COST_SCALE         : {TIME_COST_SCALE}\n")
        f.write(f"  CORRECT_PLACEMENT_BONUS : {CORRECT_PLACEMENT_BONUS}\n")
        f.write(f"  WRONG_PLACEMENT_PENALTY : {WRONG_PLACEMENT_PENALTY}\n")
        f.write(f"  WORD_COMPLETE_BONUS     : {WORD_COMPLETE_BONUS}\n")
        f.write(f"  CURRICULUM_STAGE        : {curriculum_stage}\n")
        f.write(f"\n  -- PPO Hyperparameters --\n")
        f.write(f"  TOTAL_TIMESTEPS         : {total_timesteps}\n")
        f.write(f"  N_ENVS                  : {N_ENVS}\n")
        f.write(f"  learning_rate           : {model.learning_rate}\n")
        f.write(f"  n_steps                 : {model.n_steps}\n")
        f.write(f"  batch_size              : {model.batch_size}\n")
        f.write(f"  ent_coef                : {model.ent_coef}\n")
        f.write(f"{'='*48}\n")


# ============================================================
# Environment factory
# ============================================================
def make_env():
    """
    Instantiate WordleEnv with the reward and observation callbacks from this file.

    Keeping callbacks in train.py decouples reward shaping from the environment class —
    reward design can be iterated here without modifying env/wordle_env.py.

    Returns:
        WordleEnv configured for CURRICULUM_STAGE with custom_reward and custom_observation.
    """
    from env.wordle_env import WordleEnv
    return WordleEnv(
        stage                = CURRICULUM_STAGE,
        reward_callback      = custom_reward,
        observation_callback = custom_observation,
    )


# ============================================================
# Training entry point
# ============================================================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)

    env = make_env()

    # Resume from the latest checkpoint if one exists. This is the intended workflow
    # for advancing curriculum stages: bump CURRICULUM_STAGE above and re-run —
    # the previous stage's weights are loaded and training continues from the same
    # TensorBoard step counter (reset_num_timesteps=False below).
    latest_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    if os.path.exists(latest_path + ".zip"):
        print(f"Resuming from {latest_path}.zip  (Stage {CURRICULUM_STAGE}) ...")
        model = MaskablePPO.load(latest_path, env=env, tensorboard_log=LOGS_DIR)
    else:
        print(f"No previous model found — training from scratch (Stage {CURRICULUM_STAGE}).")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate   = LEARNING_RATE,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            ent_coef        = 0.01,   # entropy bonus coefficient — encourages exploration
            tensorboard_log = LOGS_DIR,
            verbose         = 1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
    )

    # reset_num_timesteps=False keeps the TensorBoard x-axis continuous across
    # curriculum stage transitions so all runs appear on the same chart.
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = checkpoint_callback,
        reset_num_timesteps = False,
    )

    # Write a versioned archive and overwrite _latest so the next run resumes here.
    version = get_next_version()
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{version}"))
    model.save(latest_path)
    save_training_log(version, model, model.num_timesteps, CURRICULUM_STAGE)
    print(f"Saved  ->  {MODEL_DIR}/{MODEL_NAME}_v{version}.zip  +  {latest_path}.zip")
    print(f"Log    ->  {LOG_FILE}")
