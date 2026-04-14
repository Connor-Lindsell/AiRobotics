import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
import simple_driving
import time
from train import custom_reward, custom_observation

def test_policy():
    print("Loading saved PPO model...")
    # Baseline load (we provided you with an untrained dummy baseline)
    # model = PPO.load("model/ppo_simple_driving_model", env=env)

    print("Loading environment with rendering enabled...")
    env = gym.make("SimpleDriving-v0", renders=True, isDiscrete=False, reward_callback=custom_reward, observation_callback=custom_observation)
    model = PPO.load("ppo_car", env=env)

    scenarios = ["midpoint", "none", "random_pos"]
    print(f"Starting evaluation covering the {len(scenarios)} required obstacle scenarios...")

    for ep, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {ep + 1}: {scenario.upper()} ---")
        obs, info = env.reset(options={"scenario": scenario})
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            time.sleep(0.01)
            
        print(f"Episode {ep + 1} finished - Total Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_policy()
