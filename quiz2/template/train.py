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
OBSTACLE_PENALTY = ...
GOAL_REWARD = ...
STEP_PENALTY = ...
PROGRESS_REWARD_SCALE = ...
MINIMUM_SAFE_DISTANCE = 1.0

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
    # ========================================================
    # TODO: Write your reward function
    # 1. Give the agent a basic STEP_PENALTY every frame
    # 2. Reward it for getting closer to the goal
    # 3. Give it a large GOAL_REWARD if it reached_goal
    # 4. Give it a large OBSTACLE_PENALTY if it gets too close to the obstacle
    # 
    # HINT: If your agent has trouble avoiding the obstacle and drives right into it,
    # you can try adding a "proximity penalty" (repulsive field). If the car gets 
    # within a certain distance of the obstacle, start gradually subtracting reward!
    # ========================================================
    
    reward = 0.0 # Dummy return, replace this
    return reward

# You can change these variables for more training steps or if you have a powerful CPU:
TOTAL_TIMESTEPS = 75000      # define the number of steps used during the training
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

    # ========================================================
    # TODO: Implement PPO using stable_baselines3!
    # 1. Instantiate the PPO agent ("MlpPolicy")
    #    HINT: SB3's default PPO parameters are optimized for long tasks. 
    #    For our short driving environment, training will be painfully slow
    #    unless you override these hyperparameters during instantiation:
    #      - learning_rate=0.0003
    #      - n_steps=512
    #      - batch_size=256
    #      - ent_coef=0.01
    #    You can play around with different parameters, change the number of
    #    TOTAL_TIMESTEPS, learning_rate, etc.
    # 2. Tell the agent to log metrics to a local tensorboard directory.
    # 3. Call agent.learn(total_timesteps=TOTAL_TIMESTEPS)
    # 4. Save the agent when done
    # 
    # Optional: to speed up the training and avoiding to start from scratch every time, 
    # you can reload previously trained models 
    # (look up Curriculum Learning/Transfer Learning to learn more about this)
    # 
    # If you do, keep track of the previous reward function you used for the VIVA 
    # (or retrain from scratch to make sure your function works properly)
    # ========================================================
    
    print("Dummy script - Implement PPO here.")
