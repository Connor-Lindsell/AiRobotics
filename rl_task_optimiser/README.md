# RL-Based Task Sequencer — Wordle Bot (UR3e)

Reinforcement learning module for optimal pick-and-place task sequencing on a UR3e robotic arm. The agent operates purely at the **planning level** — given a set of lettered objects and a target word, it outputs an ordered action sequence that minimises total motion time. All trajectory execution is handled downstream by MoveIt 2 Task Constructor.

> This module is one of two AI components in the Wordle Bot project. The other is a CNN-based letter classifier that provides object identities and pose estimates as state input to this agent at inference time.

---

## Table of Contents

- [System Context](#system-context)
- [MDP Formulation](#mdp-formulation)
- [Implementation Stack](#implementation-stack)
- [Curriculum Training](#curriculum-training)
- [Validation & Metrics](#validation--metrics)
- [ROS 2 Integration](#ros-2-integration)
- [Known Constraints & Risks](#known-constraints--risks)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

---

## System Context

```
Camera → CNN (letter ID + pose) → Gamification (Wordle logic) → RL Agent (sequencer) → MoveIt 2 → UR3e
```

The RL agent sits between the gamification node and the MoveIt 2 executor. It receives the current world state (object identities, poses, slot occupancy, target word) and outputs a discrete action index — which object to pick next and where to place it. It does **not** interface with joint control, trajectory generation, or motion planning.

**Scope boundary:** RL = task sequencing only. Any expansion into motion control is explicitly out of scope.

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | Remaining object identities + poses (from CNN), filled/empty slot states, target word encoding |
| **Action** | Discrete — select next `(object, target_slot)` pair |
| **Action Masking** | Invalid actions (already-placed objects, filled slots) masked **before softmax** via `MaskablePPO` — not as a post-hoc penalty |
| **Reward** | Negative per-step time cost (estimated motion duration) + large terminal bonus on success + penalty on incorrect placement |
| **Episode** | Ends on successful word placement or incorrect placement |

---

## Implementation Stack

| Component | Tool |
|-----------|------|
| Algorithm | `MaskablePPO` — [`sb3-contrib`](https://github.com/Stable-Baselines3/stable-baselines3-contrib) |
| Training Environment | Custom `gymnasium.Env` — lightweight, no physics |
| Episode Resets | Randomise object poses each episode |
| Visualisation | UR3e URDF policy rollout via PyBullet / Polyscope *(sequence animation only — no physics training)* |
| Baseline Comparator | Greedy nearest-neighbour sequencer |
| ROS 2 Interface | Lightweight Python inference node |

---

## Curriculum Training

Training uses a 4-stage curriculum to progressively increase task difficulty and build sim-to-real robustness:

| Stage | Objects | Pose Noise | Purpose |
|-------|---------|------------|---------|
| 1 | 1 | None | Learn basic pick-and-place structure |
| 2 | 2 | None | Learn pairwise sequencing |
| 3 | 5 | None | Full task, randomised poses |
| 4 | 5 | Gaussian (σ tuned to CNN error) | Sim-to-real robustness |

> **Stage 4 noise is mandatory.** CNN pose estimation error (~10–15%) is the primary source of real-world performance degradation. Training without noise produces a policy that is brittle to sensor uncertainty.

---

## Validation & Metrics

### Training Monitoring

- Episodic reward curves
- Episode length (steps to task completion)
- Policy entropy over training (convergence indicator)

### Sim-to-Real Gap Analysis

The policy is evaluated against:
1. **Ground truth poses** — upper bound on performance
2. **Live CNN pose outputs** — real-world performance

The delta between these two quantifies the sim-to-real gap introduced by CNN pose error.

### Baseline Comparison

A greedy nearest-neighbour sequencer provides a deterministic baseline. Cycle time improvement of the RL policy over this baseline is the primary task-level performance metric.

---

## ROS 2 Integration

A lightweight ROS 2 Python node runs policy inference at runtime:

```
Input:  state dictionary  {object_ids, poses, slot_states, target_word}
Output: action index      → (object, target_slot) pair
```

The resulting task plan is passed directly to MoveIt 2 Task Constructor for execution. The inference node adds no overhead to the motion pipeline — it runs once per word, not per waypoint.

```
ros2 run <package> rl_sequencer_node
```

> Subscriptions and topic names: *[fill in once ROS 2 node is finalised]*

---

## Known Constraints & Risks

| Risk | Mitigation |
|------|-----------|
| CNN pose error degrades policy performance | Gaussian noise injection in Stage 4 curriculum |
| Action masking applied post-hoc corrupts learning | `MaskablePPO` enforces masking before softmax — confirmed correct |
| Sim-to-real gap from no-physics training | Pose noise + URDF rollout validation before real deployment |
| RL scope creep into motion control | Hard scope boundary — agent outputs action index only, MoveIt 2 owns all motion |

---

## Repository Structure

```
rl_task_optimiser/
├── env/
│   └── wordle_env.py          # Custom gymnasium.Env — MDP, observation, action masking
├── models/                    # Saved policy checkpoints (versioned + _latest)
├── logs/                      # TensorBoard logs + per-scenario visualisation PNGs
├── train.py                   # MaskablePPO training entry point, reward/obs callbacks
├── test.py                    # Evaluation loop, greedy baseline, matplotlib visualisation
└── README.md
```

---

## Dependencies

All dependencies are pure Python — no ROS 2 required to train or evaluate. ROS 2 (Humble or later) is only needed to run the inference node at deployment time.

| Package | Version | Purpose |
|---------|---------|---------|
| `Python` | ≥ 3.9 | Runtime |
| `gymnasium` | ≥ 0.29 | Environment base class |
| `stable-baselines3` | ≥ 2.3 | PPO implementation, callbacks |
| `sb3-contrib` | ≥ 2.3 | `MaskablePPO` (action masking before softmax) |
| `torch` | ≥ 2.0 | Neural network backend for SB3 |
| `numpy` | ≥ 1.24 | Observation arrays, pose sampling |
| `matplotlib` | ≥ 3.7 | Test visualisation (workspace plots, reward curves) |

> `stable-baselines3` and `sb3-contrib` must be the **same minor version** — mismatches cause silent import errors.

### Environment Setup

It is strongly recommended to use a Python virtual environment to avoid dependency conflicts.

**Create and activate a virtual environment:**

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**Install dependencies:**

CPU (all platforms):
```bash
pip install gymnasium "stable-baselines3[extra]>=2.3" "sb3-contrib>=2.3" numpy matplotlib
```

GPU (NVIDIA only) — install PyTorch with CUDA before the rest:
```bash
# Check pytorch.org for the exact command matching your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium "stable-baselines3[extra]>=2.3" "sb3-contrib>=2.3" numpy matplotlib
```

**Verify the critical dependency:**
```bash
python -c "from sb3_contrib import MaskablePPO; print('MaskablePPO OK')"
```

---

## How to Run

### 1. Train the agent

Training runs from `train.py`. The curriculum stage is set by the `CURRICULUM_STAGE` constant at the top of the file (default: `1`).

Linux / macOS:
```bash
python train.py
```

Windows:
```cmd
python train.py
```

On first run this creates `models/` and `logs/` directories. Periodic checkpoints are saved every `SAVE_FREQ` steps; the final model is saved as both a versioned file (`wordle_ppo_v1.zip`) and `wordle_ppo_latest.zip`.

**Advancing through curriculum stages:**

1. Open `train.py` and increment `CURRICULUM_STAGE` (e.g. `1` → `2`).
2. Re-run `python train.py` — the previous stage's `_latest` checkpoint is loaded automatically and training continues from where it left off.
3. Repeat for stages 3 and 4.

> Stage 4 adds Gaussian pose noise to simulate CNN estimation error. Do not skip it before deploying to the real robot.

**Monitor training with TensorBoard:**

Open a second terminal in the same directory while training is running (or after):

Linux / macOS:
```bash
tensorboard --logdir logs/
```

Windows:
```cmd
tensorboard --logdir logs\
```

Then open `http://localhost:6006` in a browser. Key metrics: `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `train/entropy_loss`.

If running on a remote machine, expose TensorBoard over the network:

Linux / macOS:
```bash
tensorboard --logdir logs/ --host 0.0.0.0
```

Windows:
```cmd
tensorboard --logdir logs\ --host 0.0.0.0
```

---

### 2. Evaluate the trained policy

Evaluation runs from `test.py`. It loads `models/wordle_ppo_latest.zip`, runs the RL agent and a greedy nearest-neighbour baseline across all named scenarios, prints a per-scenario summary, and saves a 4-panel visualisation PNG to `logs/` for each scenario.

Linux / macOS:
```bash
python test.py
```

Windows:
```cmd
python test.py
```

**Example output:**
```
Loading saved MaskablePPO model...
Starting evaluation across 4 scenarios...

--- Scenario 1: STAGE1_CLEAN ---
    Stage 1 — single object, no pose noise
RL Agent        — Total Reward: 48.31  Steps: 1  Success: ✓
Greedy Baseline — Total Reward: 47.12  Steps: 1  Success: ✓
RL vs Greedy delta: +1.19
Figure saved -> logs/stage1_clean_visualisation.png
```

Visualisation figures show:
- **Top row:** 2D workspace map with object positions, slot boxes, and pick-and-place arrows (RL agent left, greedy baseline right)
- **Bottom left:** Cumulative reward curves for both policies overlaid
- **Bottom right:** Action sequence timeline — which object was placed in which slot at each step

---

### 3. Quick sanity check (no model required)

To verify the environment runs correctly without a trained model:

Linux / macOS:
```bash
python -c "
from env.wordle_env import WordleEnv
from train import custom_reward, custom_observation
env = WordleEnv(stage=1, reward_callback=custom_reward, observation_callback=custom_observation)
obs, _ = env.reset()
print('Obs shape:', obs.shape)
print('Valid actions:', env.action_masks().sum())
env.render()
"
```

Windows:
```cmd
python -c "from env.wordle_env import WordleEnv; from train import custom_reward, custom_observation; env = WordleEnv(stage=1, reward_callback=custom_reward, observation_callback=custom_observation); obs, _ = env.reset(); print('Obs shape:', obs.shape); print('Valid actions:', env.action_masks().sum()); env.render()"
```

Or save the snippet to a file and run it:
```cmd
python sanity_check.py
```


