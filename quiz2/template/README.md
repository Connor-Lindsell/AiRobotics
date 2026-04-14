# AI for Robotics- Quiz 2: Reinforcement Learning

Welcome to the quiz 2 on Reinforcement Learning! In this quiz, your goal is to train an autonomous car to navigate a continuous physical space and reach a target goal while actively avoiding a randomly spawned obstacle.

You will achieve this by building a **Reward Function**, defining a **Observation Space**, and using the **Proximal Policy Optimization (PPO)** algorithm from Stable-Baselines3 to train a Policy neural network.

## 1. Environment Setup

It is highly recommended that you train your model within a Python virtual environment to prevent dependency conflicts.

### Step 1: Create a Virtual Environment

For Linux / macOS:
```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

For Windows
```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### Step 2: Install Dependencies

Note that the training on CPU is manageable (you can train your network in 5 to 10 minutes), but if you have an NVIDIA GPU, you can use it to speed up the training.

**Option A: Standard Installation (CPU only / Mac)**
```bash
pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib
```

**Option B: training on GPU (NVIDIA only)**
If you have an NVIDIA card and want faster multi-threaded PPO training, install the CUDA version of PyTorch first, and then install the rest:
```bash
# Note: Check the exact PyTorch version for your CUDA toolkit on pytorch.org
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib
```

---

## 2. Codebase Summary

Please familiarize yourself with the assignment structure before beginning:

*   **`simple_driving/`**: This is the core physics environment built using the PyBullet engine. **You should not have to modify these files.** It constructs the chassis, handles the collision math, and runs the step simulation. There are no expectations for you to understand the inner working of this code section for the VIVA.
*   **`train.py`**: This is the file where you will do your Heavy RL Lifting! It contains empty `custom_observation()` and `custom_reward()` functions for you to implement. It is also the script where you will assemble the environment block, instantiate the PPO algorithm, and run your asynchronous training loop. We have provided configurable reward constants at the top under `math utilities` in case you want to try different strategies.
*   **`test.py`**: Use this script **after** you have a trained `.zip` model. It will boot up the PyBullet GUI, spawn your car, test it across 3 distinct boundary cases (no obstacle, obstacle in the middle, and randomly placed obstacle), and output your final scores! It might be a good idea to run it multiple times once you are done to check that you did not get lucky.

---

## 3. Tasks to Implement

To get the car to drive successfully, complete the following iterative steps:

### Task A: The Observation Feed (`train.py`) [20 points]
In `custom_observation()`, the environment hands you the raw, absolute Global Coordinates $(X, Y)$ of the car, goal, and obstacle.
1. The neural network cannot generalize absolute coordinates well.
2. Use PyBullet's built-in `client.invertTransform` and `client.multiplyTransforms` to convert the `goal_pos` and `obstacle_pos` into **Relative Coordinates** (i.e. where are they relative to the hood of my car?).
3. Pack these into a flat `[float, float, ...]` array and return them.
**CRITICAL:** Your returned observation array **must** be exactly size 5 (e.g., `[goal_X, goal_Y, obstacle_X, obstacle_Y, has_obstacle]`), or PPO will crash. If there is no obstacle spawned during the episode, you must pad the remaining 3 dimensions with `0.0`! If you need a larger observation space, you will have to modify the files in `simple_driving`.

### Task B: The Reward Landscape (`train.py`) [30 points]
In `custom_reward()`, you must shape the behavior of the agent by calculating **a single scalar** `float` reward for every physical frame.
1. Apply a `STEP_PENALTY` on every frame to encourage urgency (so the car doesn't spin in circles safely).
2. Add a reward that quantify the progress made towards the goal.
3. Add a large reward if the agent `reached_goal`.
4. If `has_obstacle` is True, add a large penalty if the car colides with the obstacle.

### Task C: The Training Loop (`train.py`) [20 points]
1. Import `PPO` and `make_vec_env` from stable-baselines3.
2. Initialize multiple CPU environments using `SubprocVecEnv` (this allows parallel car training which is exponentially faster). Ensure you pass your `custom_reward` and `custom_observation` through the `env_kwargs` (environment keyword arguments).
3. Instantiate the `PPO` algorithm. (Be sure to set `tensorboard_log="./ppo_tensorboard/"`).
4. Call `.learn(total_timesteps=...)` and `.save()` the `.zip` archive!

### Task D: Iteration & Curriculum Learning (`train.py`) [10 points]
The secret to Reinforcement Learning is to not start from scratch every time you make a tweak.
If your car learns to drive straight but crashes into obstacles occasionally, **don't** destroy the brain and start over! Use **Curriculum Learning**:
* Use `model = PPO.load("my_previous_model", env=env)` inside your `train.py`.
* Continue `.learn()` so that the network only has to learn obstacle avoidance, rather than re-learning how to steer!

### Task E: Visualisation with Tensorboard [20 points]

An RL agent is a black box, and watching print logs is a terrible way to know if it's actually working. **Tensorboard** allows you to plot the average rewards per episode alongside other metrics.

1. Ensure your PPO initialization in `train.py` has `tensorboard_log="./ppo_tensorboard/"`.
2. While `train.py` is running, open a completely separate terminal.
3. Activate your virtual environment in that terminal.
4. Run the following command:
   ```bash
   tensorboard --logdir=./ppo_tensorboard/
   ```
5. It will give you a local URL (usually `http://localhost:6006/`). Open this in your web browser!

**What to look for?**
Under the `rollout/` graph section, watch the `ep_rew_mean` (Episode Reward Mean) graph. If your reward math is correct, this graph should steadily climb upwards from negative numbers and successfully converge into the positives as it learns to reach the target!


### Task F: Testing (`test.py`) [0 points]
To validate that your model has learned properly, test the trained model using `test.py`. Once you think you are done with the training, you might want to run it several times to make sure that your model is robust.

This task has no points awarded but will be critical to evaluate your success for the quiz.

---

## Submission

For your submission, you will have to submit the following files in a zip file (see how to create a zip file [here](https://copyrightservice.co.uk/reg/creating-zip-files)):
- `train.py`
- `test.py`
- your model in a `.zip` format (please only submit a single model, not all the models you tried)
- the `simple_driving` folder if you modified any component there (include it if you are not sure)
- a screenshot of the Tensorboard visualisation (only display the relevant plots, your screenshot should show a successful training of your model, i.e. the reward should be positive and increasing)
