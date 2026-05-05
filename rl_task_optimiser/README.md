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
- [Usage](#usage)

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
rl_sequencer/
├── env/
│   ├── wordle_env.py          # Custom gymnasium.Env
│   └── curriculum.py          # Stage scheduler
├── training/
│   ├── train.py               # MaskablePPO training entry point
│   └── callbacks.py           # Reward/entropy logging
├── inference/
│   └── rl_sequencer_node.py   # ROS 2 inference node
├── eval/
│   ├── sim_to_real.py         # Gap analysis script
│   └── baseline_greedy.py     # Nearest-neighbour comparator
├── models/                    # Saved policy checkpoints
├── logs/                      # TensorBoard training logs
└── README.md
```

> *Structure subject to change as implementation progresses.*

---

## Dependencies

```bash
pip install stable-baselines3 sb3-contrib gymnasium torch numpy
```

ROS 2 (Humble or later) required for the inference node. MoveIt 2 handles all downstream execution — no direct dependency from this module.

```bash
# Verify MaskablePPO is available
python -c "from sb3_contrib import MaskablePPO; print('ok')"
```

---

## Environment Setup

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



