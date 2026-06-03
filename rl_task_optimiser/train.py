import os
from collections import deque

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# ============================================================
# PPO Hyperparameters
# ============================================================
# Total environment steps per training run. Later curriculum stages need more
# steps to converge — increase if mean episode reward has not plateaued.
TOTAL_TIMESTEPS = 400_000

# Adam optimizer step size. Reduced from 3e-4 — at convergence a lower rate
# stabilises policy updates and reduces reward variance.
LEARNING_RATE   = 5e-5

# Steps collected per environment before each gradient update.
# Increased from 4096 — larger rollouts average out stochastic episode variance.
N_STEPS         = 4096

# Minibatch size. Increased from 128 for smoother gradient estimates.
# Must divide N_STEPS evenly (8192 / 256 = 32, ratio unchanged).
BATCH_SIZE      = 256

# Save a checkpoint every SAVE_FREQ environment steps.
SAVE_FREQ       = 100_000

ENTROPY_COEFFICIENT = 0.001

# ============================================================
# Reward shaping parameters
# ============================================================
# Completing all required slots dominates travel-distance optimisation.
# Correctness rewards are at least 20× larger than expected travel penalties.

WORD_COMPLETE_BONUS       = 100.0   # all required_slots correctly filled
CORRECT_PLACEMENT_BONUS   =  20.0   # per newly correct Wordle slot (once per slot)
CLEARING_BONUS            =  15.0   # clearing a wrong Wordle letter to staging
WRONG_SLOT_PENALTY        = -10.0   # letter placed in wrong Wordle slot
MOVE_CORRECT_OUT_PENALTY  = -10.0   # evicting a correctly-placed letter
STEP_PENALTY              =  -0.5   # per step — C3/C4 have variable step counts so
                                    # this provides a signal to complete efficiently
TRAVEL_COST_SCALE         =  -2.0   # × travel distance (metres) — raised from -0.5 so
                                    # travel cost is ~24% of episode reward (was ~7%)

# Reward per metre that RL beats greedy (negative when RL loses).
# At 3.0: beating greedy by 5 m earns +15; losing by 5 m costs -15.
COMPETITION_SCALE         =   3.0

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
CURRICULUM_STAGE = 3

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
# Path length tracking callback
# ============================================================
class PathLengthCallback(BaseCallback):
    """
    Tracks cumulative_travel (metres) per episode and exposes ep_path_length_mean.

    SB3's built-in ep_info_buffer only captures 'r' and 'l'. This callback
    reads info["cumulative_travel"] from the env at episode end and maintains
    its own rolling buffer of the same size so the metric can be logged and
    compared to ep_len_mean.
    """

    def __init__(self, buffer_size: int = 100):
        super().__init__()
        self._path_buffer: deque[float] = deque(maxlen=buffer_size)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                travel = self.locals["infos"][i].get("cumulative_travel", 0.0)
                self._path_buffer.append(travel)
        if self._path_buffer:
            mean_path = float(np.mean(self._path_buffer))
            self.model.logger.record("rollout/ep_path_length_mean", mean_path)
        return True

    @property
    def ep_path_length_mean(self) -> float | None:
        return float(np.mean(self._path_buffer)) if self._path_buffer else None


# ============================================================
# Greedy competition wrapper
# ============================================================
class GreedyCompetitionWrapper(gym.Wrapper):
    """
    After every completed episode, adds COMPETITION_SCALE × (greedy_travel − rl_travel)
    to the final step reward.  Positive when RL beats greedy, negative when it loses.

    On each reset() the wrapper clones the fresh board state onto a shadow env and
    runs the greedy solver (min trip-cost at each step) to record the baseline
    travel distance for that specific task instance.
    """

    def __init__(self, env):
        super().__init__(env)
        from training_env.wordle_env import WordleEnv
        self._shadow = WordleEnv(
            stage           = env.stage,
            reward_callback = env.reward_callback,
        )
        self._greedy_travel: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._greedy_travel = self._run_greedy()
        return obs, info

    def _run_greedy(self):
        from training_env.wordle_env import ALL_POSITIONS, N_CELLS, compute_travel
        src = self.env
        s   = self._shadow

        # Clone board state — no randomisation, exact copy of what RL will face
        s.target_word            = src.target_word
        s.position_letter        = list(src.position_letter)
        s.position_occupied      = src.position_occupied.copy()
        s.wordle_correct         = src.wordle_correct.copy()
        s.robot_pos              = src.robot_pos.copy()
        s.required_slots         = set(src.required_slots)
        s.already_rewarded_slots = set()
        s._step_count            = 0
        s._cumulative_travel     = 0.0
        s._invalid_action_count  = 0
        s.action_log             = []

        done = False
        while not done:
            masks = s.action_masks()
            valid = [i for i, m in enumerate(masks) if m]
            if not valid:
                break
            best = min(
                valid,
                key=lambda a: compute_travel(
                    s.robot_pos,
                    ALL_POSITIONS[a // N_CELLS],
                    ALL_POSITIONS[a %  N_CELLS],
                ),
            )
            _, _, terminated, truncated, _ = s.step(best)
            done = terminated or truncated

        return s._cumulative_travel

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and info.get("word_complete"):
            reward += COMPETITION_SCALE * (self._greedy_travel - info["cumulative_travel"])
        return obs, reward, terminated, truncated, info


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

    if clearing_to_staging and not moving_correct_out:
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
    env = WordleEnv(
        stage            = CURRICULUM_STAGE,
        reward_callback  = custom_reward,
        observation_callback = None,
    )
    return GreedyCompetitionWrapper(env)


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


def save_training_log(version, model, total_timesteps, curriculum_stage, path_cb: PathLengthCallback | None = None):
    """Append a training run summary to LOG_FILE."""
    from datetime import datetime
    buf        = model.ep_info_buffer
    ep_rew     = round(float(np.mean([e["r"] for e in buf])), 2) if buf else "N/A"
    ep_len     = round(float(np.mean([e["l"] for e in buf])), 2) if buf else "N/A"
    ep_path    = round(path_cb.ep_path_length_mean, 2) if path_cb and path_cb.ep_path_length_mean is not None else "N/A"

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*48}\n")
        f.write(f"Run v{version}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Stage C{curriculum_stage}\n")
        f.write(f"\n  -- Results --\n")
        f.write(f"  total_timesteps         : {total_timesteps}\n")
        f.write(f"  ep_rew_mean             : {ep_rew}\n")
        f.write(f"  ep_len_mean             : {ep_len}\n")
        f.write(f"  ep_path_length_mean (m) : {ep_path}\n")
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
        # custom_objects only handles non-serialisable classes — set scalars directly
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = LEARNING_RATE
        model.lr_schedule   = get_schedule_fn(LEARNING_RATE)
        model.clip_range    = get_schedule_fn(0.2)
        model.ent_coef      = ENTROPY_COEFFICIENT
        model.n_steps       = N_STEPS
        model.batch_size    = BATCH_SIZE
        model.n_epochs      = 5
        model.vf_coef       = 1.0   # boosted: critic value_loss is in the hundreds, needs more gradient
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = LEARNING_RATE
        print(f"  Hyperparameters overridden: lr={LEARNING_RATE}, ent_coef={ENTROPY_COEFFICIENT}, vf_coef=1.0")
    else:
        print(f"No previous model found — training from scratch (Stage C{CURRICULUM_STAGE}).")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate   = LEARNING_RATE,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = 5,      # fewer epochs reduces over-optimisation on stale rollouts
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.1,    # tighter than 0.2 — smaller policy steps at convergence
            ent_coef        = ENTROPY_COEFFICIENT,  # low entropy: policy is converged, exploration adds noise
            vf_coef         = 1.0,    # boosted: critic value_loss in the hundreds, needs more gradient
            tensorboard_log = LOGS_DIR,
            verbose         = 1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
    )
    path_callback = PathLengthCallback(buffer_size=100)

    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = [checkpoint_callback, path_callback],
        reset_num_timesteps = False,
    )

    version = get_next_version()
    model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_v{version}"))
    model.save(latest_path)
    save_training_log(version, model, model.num_timesteps, CURRICULUM_STAGE, path_callback)
    print(f"Saved  ->  {MODEL_DIR}/{MODEL_NAME}_v{version}.zip  +  {latest_path}.zip")
    print(f"Log    ->  {LOG_FILE}")
