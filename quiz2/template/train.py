import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import simple_driving
import time
import os
import math

# ========================================================
# Reward Function Configuration Parameters
# ========================================================
OBSTACLE_PENALTY = -400.0
GOAL_REWARD = 900.0
STEP_PENALTY = -2.0
PROGRESS_REWARD_SCALE = 10.0
MINIMUM_SAFE_DISTANCE = 1.0
PROXIMITY_SCALE          = 0.25 
OBSTACLE_PROXIMITY_SCALE = 2.0  

MODEL_DIR  = "model"
MODEL_NAME = "ppo_car"
LOG_FILE   = "training_log.txt"

def custom_observation(client, car_pos, car_orn, goal_pos, goal_orn, obstacle_pos, has_obstacle):
    """
    Computes the observation array for the neural network.
    
    Args:
        client (bullet_client): The PyBullet physics client.
        car_pos (list of float): The global [x, y, z] position of the car.
        car_orn (list of float): The global [x, y, z, w] quaternion orientation of the car.
        goal_pos (list of float): The global [x, y, z] position of the goal.
        goal_orn (list of float): The global [x, y, z, w] quaternion orientation of the goal.
        obstacle_pos (tuple of float or None): The global (x, y) position of the obstacle, if it exists.
        has_obstacle (bool): True if an obstacle spawned this episode, False otherwise.
        
    Returns:
        list of float: The computed observation state array.
    """
    # Step 1: Invert the car's world transform to get the world → car-frame transform
    inv_car_pos, inv_car_orn = client.invertTransform(car_pos, car_orn)

    # Step 2: Express the goal position in the car's local frame
    goal_rel_pos, _ = client.multiplyTransforms(
        inv_car_pos, inv_car_orn,
        goal_pos, [0, 0, 0, 1]  # identity orientation — we only care about position
    )

    # Step 3: Express the obstacle position in the car's local frame (or pad with zeros)
    if has_obstacle and obstacle_pos is not None:
        obs_world = [obstacle_pos[0], obstacle_pos[1], 0.0]  # promote (x,y) → (x,y,z)
        obs_rel_pos, _ = client.multiplyTransforms(
            inv_car_pos, inv_car_orn,
            obs_world, [0, 0, 0, 1]
        )
        obs_x, obs_y = obs_rel_pos[0], obs_rel_pos[1]
    else:
        obs_x, obs_y = 0.0, 0.0

    # Step 4: Pack into a flat size-5 array
    observation = [
        goal_rel_pos[0],      # goal X relative to car
        goal_rel_pos[1],      # goal Y relative to car
        obs_x,                # obstacle X relative to car (0 if absent)
        obs_y,                # obstacle Y relative to car (0 if absent)
        float(has_obstacle),  # 1.0 = obstacle present, 0.0 = no obstacle
    ]

    # Print Debug Info
    # print("\n======= Custom Observation Debug Info =======\n")
    # print(f"Car Pos: {car_pos}, Car Orientation: {car_orn}")
    # print(f"Goal Pos: {goal_pos}, Goal Orientation: {goal_orn}")
    # print(f"Obstacle Pos: {obstacle_pos}, Has Obstacle: {has_obstacle}")
    # print(f"Computed Observation: {observation}\n")
    # print("\n============================================\n")

    return observation


def custom_reward(car_pos, goal_pos, obstacle_pos, has_obstacle, prev_dist_to_goal, dist_to_goal, reached_goal):
    """
    Computes the scalar reward for the current timestep.
    
    Args:
        car_pos (list of float): The global [x, y, z] position of the car.
        goal_pos (list of float): The global [x, y, z] position of the goal.
        obstacle_pos (tuple of float or None): The global (x, y) position of the obstacle, if it exists.
        has_obstacle (bool): True if an obstacle spawned this episode.
        prev_dist_to_goal (float): The distance to the goal in the previous physics frame.
        dist_to_goal (float): The distance to the goal in the current physics frame.
        reached_goal (bool): True if the car reached the goal this frame.
        
    Returns:
        float: The exact mathematical reward for this timestep.
    """

    # 1. Step penalty every frame — encourages urgency, punishes spinning in circles
    reward = STEP_PENALTY

    # 2. Progress reward — asymmetric: drifting away penalised 1.5× harder than closing in is rewarded.
    progress = prev_dist_to_goal - dist_to_goal
    if progress >= 0:
        reward += progress * PROGRESS_REWARD_SCALE

    # 3. Dense proximity bonus — reward being CLOSE to goal at every step, not just at termination.
    reward += PROXIMITY_SCALE / (dist_to_goal + 1.0)

    # 4. Large bonus for reaching the goal
    if reached_goal:
        reward += GOAL_REWARD

    # 5. Obstacle collision penalty — triggered when the car enters the danger radius
    if has_obstacle and obstacle_pos is not None:
        dist_to_obstacle = math.sqrt(
            (car_pos[0] - obstacle_pos[0]) ** 2 +
            (car_pos[1] - obstacle_pos[1]) ** 2
        )
        if dist_to_obstacle < MINIMUM_SAFE_DISTANCE:
            reward += OBSTACLE_PENALTY
    
        # Penalty for to close to obstacle (but not colliding) — encourages learning to steer around it rather than just crashing through
        proximity_zone = MINIMUM_SAFE_DISTANCE * OBSTACLE_PROXIMITY_SCALE
        if dist_to_obstacle < proximity_zone:
            # Linear scale: 0 penalty at zone edge, 0.5 * OBSTACLE_PENALTY at collision boundary
            penetration = (proximity_zone - dist_to_obstacle) / (proximity_zone - MINIMUM_SAFE_DISTANCE)
            reward += OBSTACLE_PENALTY * 0.5 * penetration

    return reward

def get_next_version():
    """Find the next version number for model saving."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    v = 1
    while os.path.exists(f"{MODEL_DIR}/{MODEL_NAME}_v{v}.zip"):
        v += 1
    return v

def save_training_log(version, model, total_timesteps):
    """Append a training run summary to training_log.txt."""
    from datetime import datetime
    import numpy as np
    buf = model.ep_info_buffer
    ep_rew = round(float(np.mean([e['r'] for e in buf])), 2) if buf else "N/A"
    ep_len = round(float(np.mean([e['l'] for e in buf])), 2) if buf else "N/A"
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*48}\n")
        f.write(f"Run v{version}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n  -- Results --\n")
        f.write(f"  total_timesteps      : {total_timesteps}\n")
        f.write(f"  ep_rew_mean          : {ep_rew}\n")
        f.write(f"  ep_len_mean          : {ep_len}\n")
        f.write(f"\n  -- Reward Config --\n")
        f.write(f"  GOAL_REWARD          : {GOAL_REWARD}\n")
        f.write(f"  OBSTACLE_PENALTY     : {OBSTACLE_PENALTY}\n")
        f.write(f"  STEP_PENALTY         : {STEP_PENALTY}\n")
        f.write(f"  PROGRESS_REWARD_SCALE: {PROGRESS_REWARD_SCALE}\n")
        f.write(f"  MINIMUM_SAFE_DISTANCE: {MINIMUM_SAFE_DISTANCE}\n")
        f.write(f"  PROXIMITY_SCALE      : {PROXIMITY_SCALE}\n")
        f.write(f"  OBSTACLE_PROXIMITY_SCALE: {OBSTACLE_PROXIMITY_SCALE}\n")
        f.write(f"\n  -- PPO Hyperparameters --\n")
        f.write(f"  TOTAL_TIMESTEPS      : {TOTAL_TIMESTEPS}\n")
        f.write(f"  N_ENVS               : {N_ENVS}\n")
        f.write(f"  learning_rate        : {model.learning_rate}\n")
        f.write(f"  n_steps              : {model.n_steps}\n")
        f.write(f"  batch_size           : {model.batch_size}\n")
        f.write(f"  ent_coef             : {model.ent_coef}\n")
        f.write(f"{'='*48}\n")

# You can change these variables for more training steps or if you have a powerful CPU:
TOTAL_TIMESTEPS = 100000      # define the number of steps used during the training
N_ENVS = 4                   # number of processor core used for multithreading

if __name__ == "__main__":
    env_kwargs = {
        "renders": False, 
        "isDiscrete": False,
        "reward_callback": custom_reward,          
        "observation_callback": custom_observation 
    }
    env = make_vec_env(
        "SimpleDriving-v0", 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=env_kwargs,
        vec_env_kwargs={"start_method": "spawn"}
    )

    # Load latest model if one exists (curriculum learning), otherwise train from scratch
    latest = f"{MODEL_DIR}/{MODEL_NAME}_latest"
    if os.path.exists(latest + ".zip"):
        print(f"Resuming from {latest}.zip ...")
        model = PPO.load(latest, env=env, tensorboard_log="./ppo_tensorboard/")
    else:
        print("No previous model found — training from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = 0.0003,  #
            n_steps = 512,          
            batch_size = 256,
            ent_coef = 0.01,        # was 0.005 — less random action noise in steering
            tensorboard_log = "./ppo_tensorboard/",
            verbose = 1,
        )

    # Train — reset_num_timesteps=False keeps the step counter continuous across runs
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)

    # Save versioned copy + overwrite latest
    version   = get_next_version()
    versioned = f"{MODEL_DIR}/{MODEL_NAME}_v{version}"
    latest    = f"{MODEL_DIR}/{MODEL_NAME}_latest"
    model.save(versioned)
    model.save(latest)
    save_training_log(version, model, model.num_timesteps)
    print(f"Saved  ->  {versioned}.zip  +  {latest}.zip")
    print(f"Log    ->  {LOG_FILE}")
