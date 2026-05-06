import os
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ============================================================
# PPO Hyperparameters
# ============================================================
# Total environment steps per training run. Later curriculum stages need more
# steps to converge — increase if mean episode reward has not plateaued.
TOTAL_TIMESTEPS = 100_000

# Adam optimizer step size. 3e-4 is a reliable default for MaskablePPO.
LEARNING_RATE   = 3e-4

# Steps collected per environment before each gradient update.
N_STEPS         = 2048

# Minibatch size. Must divide N_STEPS evenly.
BATCH_SIZE      = 64

# Save a checkpoint every SAVE_FREQ environment steps.
SAVE_FREQ       = 10_000

# ============================================================
# Reward shaping parameters
# ============================================================
# Completing all required slots dominates travel-distance optimisation.
# Correctness rewards are at least 20× larger than expected travel penalties.

WORD_COMPLETE_BONUS       = 100.0   # all required_slots correctly filled
CORRECT_PLACEMENT_BONUS   =  20.0   # per newly correct Wordle slot (once per slot)
CLEARING_BONUS            =   5.0   # clearing a wrong Wordle letter to staging
WRONG_SLOT_PENALTY        = -20.0   # letter placed in wrong Wordle slot
MOVE_CORRECT_OUT_PENALTY  = -10.0   # evicting a correctly-placed letter
STEP_PENALTY              =  -1.0   # per symbolic step
TRAVEL_COST_SCALE         =  -0.05  # × travel distance (metres) — kept small

# ============================================================
# Curriculum stage
# ============================================================
# Active stage (1–5). Advance manually between runs: bump this value and re-run.
# The script auto-resumes from the latest checkpoint so learned weights transfer
# across all stages (obs and action shapes are identical for C1–C5).
#
#   C1: 1 letter  → 1 staging slot occupied
#   C2: 3 letters → 3 staging slots occupied
#   C3: 5 letters → 5 staging slots occupied, full word
#   C4: 1 wrong Wordle letter + 5 correct letters in staging (clear + fill)
#   C5: 3–5 wrong Wordle letters + 5 correct letters in staging (full rearrange)
CURRICULUM_STAGE = 1

# ============================================================
# Model and log paths
# ============================================================
MODEL_DIR  = "models"
MODEL_NAME = "wordle_ppo"
LOGS_DIR   = "logs"
LOG_FILE   = "training_log.txt"

# MaskablePPO requires action_masks() to be called synchronously on the env.
N_ENVS = 1


# ============================================================
# Reward callback
# ============================================================
def custom_reward(
    placing_correct:       bool,
    placing_wrong_wordle:  bool,
    clearing_to_staging:   bool,
    moving_correct_out:    bool,
    word_complete:         bool,
    travel_distance:       float,
    slot_already_rewarded: bool,
    **kwargs,   # source_is_wordle, dest_is_wordle, source_is_staging, dest_is_staging, step_count, …
) -> float:
    """
    Layered reward for a single symbolic pick-and-place action.

    Correct task completion is designed to dominate travel-distance optimisation:
    - Completing the word alone is worth +100.
    - Each correct placement is +20.
    - Maximum travel cost per step (across the full workspace diagonal ~28 m)
      is ≈ 1.4, well below the placement bonus.

    Args:
        placing_correct       : letter matches target at dest Wordle slot
        placing_wrong_wordle  : letter does NOT match target at dest Wordle slot
        clearing_to_staging   : moving a letter out of a Wordle slot to staging
        moving_correct_out    : evicting a correctly-placed letter from its slot
        word_complete         : all required slots are now correctly filled
        travel_distance       : robot → source → dest, metres
        slot_already_rewarded : dest slot already received correct-placement bonus
        **kwargs              : extra context from the env (source_is_wordle,
                                dest_is_wordle, source_is_staging, dest_is_staging,
                                step_count) — available for custom reward extensions

    Returns:
        float: scalar reward for this step
    """
    reward = STEP_PENALTY
    reward += TRAVEL_COST_SCALE * travel_distance

    if moving_correct_out:
        reward += MOVE_CORRECT_OUT_PENALTY

    if clearing_to_staging:
        reward += CLEARING_BONUS

    if placing_correct and not slot_already_rewarded:
        reward += CORRECT_PLACEMENT_BONUS

    if placing_wrong_wordle:
        reward += WRONG_SLOT_PENALTY

    if word_complete:
        reward += WORD_COMPLETE_BONUS

    return reward


# ============================================================
# Environment factory
# ============================================================
def make_env():
    """
    Instantiate WordleSequencingEnv for the active CURRICULUM_STAGE.

    The env builds its own observation via _build_obs(); reward logic lives
    here in custom_reward() so constants can be tuned without editing the env.
    """
    from training_env.wordle_env import WordleEnv
    return WordleEnv(
        stage            = CURRICULUM_STAGE,
        reward_callback  = custom_reward,
        observation_callback = None,   # env uses internal _build_obs()
    )


# ============================================================
# Versioned model saving helpers
# ============================================================
def get_next_version() -> int:
    """Return the lowest unused version number, scanning MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    v = 1
    while os.path.exists(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{v}.zip")):
        v += 1
    return v


def save_training_log(version, model, total_timesteps, curriculum_stage):
    """Append a training run summary to LOG_FILE."""
    from datetime import datetime
    buf    = model.ep_info_buffer
    ep_rew = round(float(np.mean([e["r"] for e in buf])), 2) if buf else "N/A"
    ep_len = round(float(np.mean([e["l"] for e in buf])), 2) if buf else "N/A"

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*48}\n")
        f.write(f"Run v{version}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Stage C{curriculum_stage}\n")
        f.write(f"\n  -- Results --\n")
        f.write(f"  total_timesteps         : {total_timesteps}\n")
        f.write(f"  ep_rew_mean             : {ep_rew}\n")
        f.write(f"  ep_len_mean             : {ep_len}\n")
        f.write(f"\n  -- Reward Config --\n")
        f.write(f"  WORD_COMPLETE_BONUS     : {WORD_COMPLETE_BONUS}\n")
        f.write(f"  CORRECT_PLACEMENT_BONUS : {CORRECT_PLACEMENT_BONUS}\n")
        f.write(f"  CLEARING_BONUS          : {CLEARING_BONUS}\n")
        f.write(f"  WRONG_SLOT_PENALTY      : {WRONG_SLOT_PENALTY}\n")
        f.write(f"  MOVE_CORRECT_OUT_PENALTY: {MOVE_CORRECT_OUT_PENALTY}\n")
        f.write(f"  STEP_PENALTY            : {STEP_PENALTY}\n")
        f.write(f"  TRAVEL_COST_SCALE       : {TRAVEL_COST_SCALE}\n")
        f.write(f"  CURRICULUM_STAGE        : C{curriculum_stage}\n")
        f.write(f"\n  -- PPO Hyperparameters --\n")
        f.write(f"  TOTAL_TIMESTEPS         : {total_timesteps}\n")
        f.write(f"  N_ENVS                  : {N_ENVS}\n")
        f.write(f"  learning_rate           : {model.learning_rate}\n")
        f.write(f"  n_steps                 : {model.n_steps}\n")
        f.write(f"  batch_size              : {model.batch_size}\n")
        f.write(f"  ent_coef                : {model.ent_coef}\n")
        f.write(f"{'='*48}\n")


# ============================================================
# Training entry point
# ============================================================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)

    env = make_env()

    # Resume from the latest checkpoint when advancing curriculum stages.
    # Obs and action shapes are identical across C1–C5, so weights transfer directly.
    latest_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    if os.path.exists(latest_path + ".zip"):
        print(f"Resuming from {latest_path}.zip  (Stage C{CURRICULUM_STAGE}) ...")
        model = MaskablePPO.load(latest_path, env=env, tensorboard_log=LOGS_DIR)
    else:
        print(f"No previous model found — training from scratch (Stage C{CURRICULUM_STAGE}).")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate   = LEARNING_RATE,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            ent_coef        = 0.01,
            tensorboard_log = LOGS_DIR,
            verbose         = 1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
    )

    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = checkpoint_callback,
        reset_num_timesteps = False,
    )

    version = get_next_version()
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{version}"))
    model.save(latest_path)
    save_training_log(version, model, model.num_timesteps, CURRICULUM_STAGE)
    print(f"Saved  ->  {MODEL_DIR}/{MODEL_NAME}_v{version}.zip  +  {latest_path}.zip")
    print(f"Log    ->  {LOG_FILE}")
