import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sb3_contrib import MaskablePPO
from training_env.wordle_env import (
    WordleEnv, CURRICULUM_STAGES, WORD_LENGTH, MAX_OBJECTS,
    SLOT_POSITIONS, WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
)
from train import custom_reward, custom_observation, MODEL_DIR, MODEL_NAME, ARM_HOME_POS, LOGS_DIR

# ============================================================
# Test configuration
# ============================================================
# Delay between steps — gives a "live" feel when watching the episode unfold.
# Set to 0.0 to run at full speed (useful for batch evaluation).
# Mirrors quiz2's time.sleep() between steps.
RENDER_DELAY = 0.05

# Named test scenarios — analogous to quiz2's ["midpoint", "none", "random_pos"].
# Each scenario creates a WordleEnv with specific parameters so results are comparable.
# TODO: Adjust target_word and pose_noise_std once CURRICULUM_STAGES are tuned.
SCENARIOS = [
    {
        "name":           "stage1_clean",
        "stage":          1,
        "target_word":    "CRANE",
        "pose_noise_std": 0.0,
        "description":    "Stage 1 — single object, no pose noise",
    },
    {
        "name":           "stage3_clean",
        "stage":          3,
        "target_word":    "CRANE",
        "pose_noise_std": 0.0,
        "description":    "Stage 3 — full word, clean poses",
    },
    {
        "name":           "stage4_noisy",
        "stage":          4,
        "target_word":    "CRANE",
        "pose_noise_std": 0.02,
        "description":    "Stage 4 — full word, Gaussian pose noise (sim-to-real)",
    },
    {
        "name":           "stage3_random_word",
        "stage":          3,
        "target_word":    None,
        "pose_noise_std": 0.0,
        "description":    "Stage 3 — random target word each run",
    },
]


# ============================================================
# Episode runners
# ============================================================

def run_episode(model, env) -> dict:
    """
    Run a single episode with the loaded MaskablePPO model.

    Uses action_masks() at each step so the model never selects an invalid action.
    Mirrors quiz2's test loop: predict → step → sleep → repeat.

    Returns:
        dict with keys:
            "action_history"     (list of (object_idx, slot_idx)): sequence of placements
            "object_poses"       (np.ndarray shape (n_objects, 2)): initial object positions
            "object_letters"     (list[str]): letter per object at episode start
            "rewards"            (list[float]): per-step reward
            "cumulative_rewards" (list[float]): cumulative reward after each step
            "target_word"        (str): the word being spelled
            "success"            (bool): True if word was completed correctly

    TODO: obs, info = env.reset()
    TODO: Snapshot env.object_poses and env.object_letters here (before any step
          modifies them) so the visualiser has initial poses for arrow drawing.
    TODO: done = False; rewards = []; cumulative = 0.0
    TODO: while not done:
              masks  = env.action_masks()
              action, _ = model.predict(obs, deterministic=True, action_masks=masks)
              obs, reward, terminated, truncated, info = env.step(action)
              rewards.append(reward)
              cumulative += reward
              cumulative_rewards.append(cumulative)
              done = terminated or truncated
              time.sleep(RENDER_DELAY)
    TODO: success = all(env.slot_occupied) and all(
              env.placed_letters[i] == env.target_word[i] for i in range(WORD_LENGTH))
    TODO: return dict with all trajectory data
    """
    obs, _ = env.reset()
    object_poses   = env.object_poses.copy()
    object_letters = list(env.object_letters)

    done               = False
    rewards            = []
    cumulative_rewards = []
    cumulative         = 0.0

    while not done:
        masks           = env.action_masks()
        action, _       = model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        done = terminated or truncated
        time.sleep(RENDER_DELAY)

    success = all(env.slot_occupied) and all(
        env.placed_letters[i] == env.target_word[i] for i in range(WORD_LENGTH)
    )

    return {
        "action_history":     env.action_history,
        "object_poses":       object_poses,
        "object_letters":     object_letters,
        "rewards":            rewards,
        "cumulative_rewards": cumulative_rewards,
        "target_word":        env.target_word,
        "success":            success,
    }


def greedy_baseline(env, initial_obs) -> dict:
    """
    Run a greedy nearest-neighbour baseline on the same env state.

    Greedy policy: at each step, select the valid (object, slot) action that
    minimises estimated motion distance:
        dist(ARM_HOME_POS → object_pose) + dist(object_pose → slot_pose)

    This provides a deterministic lower-bound comparison for the RL agent.
    The delta in cumulative reward vs. the RL agent is the primary metric.

    Args:
        env         (WordleEnv): env already reset to the desired initial state
        initial_obs (np.ndarray): observation returned by env.reset()

    Returns:
        dict — same schema as run_episode() return value

    TODO: obs = initial_obs; done = False; rewards = []; cumulative = 0.0
    TODO: while not done:
              masks = env.action_masks()
              valid_actions = [i for i, m in enumerate(masks) if m]
              # Select action minimising motion distance
              best_action = min(valid_actions, key=lambda a: _greedy_cost(a, env))
              obs, reward, terminated, truncated, info = env.step(best_action)
              rewards.append(reward)
              cumulative += reward
              done = terminated or truncated
    TODO: return trajectory dict matching run_episode() schema
    """
    object_poses   = env.object_poses.copy()
    object_letters = list(env.object_letters)

    done               = False
    rewards            = []
    cumulative_rewards = []
    cumulative         = 0.0

    while not done:
        masks         = env.action_masks()
        valid_actions = [i for i, m in enumerate(masks) if m]
        best_action   = min(valid_actions, key=lambda a: _greedy_cost(a, env))
        _, reward, terminated, truncated, _ = env.step(best_action)
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        done = terminated or truncated

    success = all(env.slot_occupied) and all(
        env.placed_letters[i] == env.target_word[i] for i in range(WORD_LENGTH)
    )

    return {
        "action_history":     env.action_history,
        "object_poses":       object_poses,
        "object_letters":     object_letters,
        "rewards":            rewards,
        "cumulative_rewards": cumulative_rewards,
        "target_word":        env.target_word,
        "success":            success,
    }


def _greedy_cost(action: int, env: WordleEnv) -> float:
    """
    Estimate motion cost for a candidate action (used by greedy_baseline).

    Cost = dist(ARM_HOME_POS → object_pose) + dist(object_pose → slot_pose)

    TODO: object_idx = action // WORD_LENGTH
    TODO: slot_idx   = action  % WORD_LENGTH
    TODO: obj_pose   = env.object_poses[object_idx]
    TODO: slot_pose  = SLOT_POSITIONS[slot_idx]
    TODO: return np.linalg.norm(obj_pose - np.array(ARM_HOME_POS)) +
                 np.linalg.norm(obj_pose - np.array(slot_pose))
    """
    object_idx = action // WORD_LENGTH
    slot_idx   = action  % WORD_LENGTH
    obj_pose   = env.object_poses[object_idx]
    slot_pose  = SLOT_POSITIONS[slot_idx]
    return (np.linalg.norm(obj_pose - np.array(ARM_HOME_POS)) +
            np.linalg.norm(obj_pose - np.array(slot_pose)))


# ============================================================
# Visualisation helpers
# ============================================================

def plot_workspace(ax, trajectory: dict, title: str):
    """
    Draw the 2D workspace view for one episode trajectory.

    Visual elements:
        1. Object circles:
           Scatter each object's INITIAL pose as a circle labelled with its letter.
           Green fill  = object was placed in the correct slot.
           Red fill    = object was placed in the wrong slot OR not placed.
           TODO: ax.scatter(x, y, s=300, color=..., zorder=3)
                 ax.annotate(letter, (x, y), ha='center', va='center', fontweight='bold')

        2. Target word slots:
           Draw WORD_LENGTH rectangles in a row at their fixed SLOT_POSITIONS.
           Each box shows the target letter above and the placed letter (or '_') inside.
           TODO: for i, (sx, sy) in enumerate(SLOT_POSITIONS):
                     box = FancyBboxPatch((sx-0.04, sy-0.04), 0.08, 0.08, ...)
                     ax.add_patch(box)
                     ax.text(sx, sy, placed or '_', ha='center', va='center')
                     ax.text(sx, sy+0.06, target_word[i], ha='center', color='grey')

        3. Pick-and-place arrows:
           For each (object_idx, slot_idx) in action_history, draw a curved arrow
           from the object's initial pose to the corresponding slot position.
           Number each arrow with its step index (1, 2, ...) near the midpoint.
           Use FancyArrowPatch with connectionstyle="arc3,rad=0.2" for curvature.
           Alternate rad sign (+/-) between steps to avoid overlap.
           TODO: for step, (oi, si) in enumerate(trajectory["action_history"]):
                     obj_pos  = trajectory["object_poses"][oi]
                     slot_pos = SLOT_POSITIONS[si]
                     rad = 0.2 if step % 2 == 0 else -0.2
                     arrow = FancyArrowPatch(
                         obj_pos, slot_pos,
                         connectionstyle=f"arc3,rad={rad}",
                         arrowstyle="-|>", color="steelblue", lw=1.5,
                     )
                     ax.add_patch(arrow)
                     mid_x = (obj_pos[0] + slot_pos[0]) / 2
                     mid_y = (obj_pos[1] + slot_pos[1]) / 2
                     ax.text(mid_x, mid_y, str(step+1), fontsize=8, color="steelblue")

    Args:
        ax         (matplotlib.axes.Axes): axes to draw on
        trajectory (dict):                 returned by run_episode() or greedy_baseline()
        title      (str):                  subplot title (e.g. "RL Agent" or "Greedy Baseline")

    TODO: ax.set_xlim(WORKSPACE_X_MIN - 0.1, WORKSPACE_X_MAX + 0.1)
    TODO: ax.set_ylim(WORKSPACE_Y_MIN - 0.1, WORKSPACE_Y_MAX + 0.1)
    TODO: ax.set_aspect("equal")
    TODO: ax.set_title(title)
    TODO: ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    TODO: Add legend: green circle = correct, red circle = wrong/unplaced, blue arrow = action sequence
    """
    target_word    = trajectory["target_word"]
    object_poses   = trajectory["object_poses"]
    object_letters = trajectory["object_letters"]
    action_history = trajectory["action_history"]

    # Determine per-object placement correctness
    correct_set = {
        oi for oi, si in action_history
        if object_letters[oi] == target_word[si]
    }

    # 1. Object circles
    for i, (letter, pose) in enumerate(zip(object_letters, object_poses)):
        color = "green" if i in correct_set else "red"
        ax.scatter(pose[0], pose[1], s=300, color=color, zorder=3)
        ax.annotate(letter, (pose[0], pose[1]), ha="center", va="center",
                    fontweight="bold", color="white", fontsize=9, zorder=4)

    # 2. Target word slots
    placed_letters = [None] * WORD_LENGTH
    for oi, si in action_history:
        placed_letters[si] = object_letters[oi]

    for i, (sx, sy) in enumerate(SLOT_POSITIONS):
        box = FancyBboxPatch(
            (sx - 0.04, sy - 0.04), 0.08, 0.08,
            boxstyle="round,pad=0.005", linewidth=1.5,
            edgecolor="black", facecolor="lightyellow", zorder=2,
        )
        ax.add_patch(box)
        ax.text(sx, sy, placed_letters[i] or "_", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=5)
        ax.text(sx, sy + 0.06, target_word[i], ha="center", va="bottom",
                fontsize=7, color="grey", zorder=5)

    # 3. Pick-and-place arrows
    for step, (oi, si) in enumerate(action_history):
        obj_pos  = object_poses[oi]
        slot_pos = SLOT_POSITIONS[si]
        rad      = 0.2 if step % 2 == 0 else -0.2
        arrow = FancyArrowPatch(
            tuple(obj_pos), slot_pos,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", color="steelblue", lw=1.5, zorder=3,
        )
        ax.add_patch(arrow)
        mid_x = (obj_pos[0] + slot_pos[0]) / 2
        mid_y = (obj_pos[1] + slot_pos[1]) / 2
        ax.text(mid_x, mid_y, str(step + 1), fontsize=8, color="steelblue", zorder=6)

    ax.set_xlim(WORKSPACE_X_MIN - 0.1, WORKSPACE_X_MAX + 0.1)
    ax.set_ylim(WORKSPACE_Y_MIN - 0.1, WORKSPACE_Y_MAX + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(handles=[
        mpatches.Patch(color="green",    label="Correct placement"),
        mpatches.Patch(color="red",      label="Wrong / unplaced"),
        mpatches.Patch(color="steelblue", label="Action sequence"),
    ], loc="upper right", fontsize=8)


def plot_action_timeline(ax, trajectory: dict, title: str):
    """
    Draw a horizontal bar chart showing which object → which slot at each step.

    Layout:
        X-axis: step number (1, 2, ..., n_steps)
        Y-axis: slot index (0–4), labelled with target_word[slot_idx]
        Each step = one horizontal bar at y=slot_idx, spanning [step, step+1]
        Bar colour: green if placed letter matched target letter, red if wrong
        Bar label:  the object letter placed (shown inside the bar)

    This makes it easy to read the action sequence in order:
        "Step 1: placed 'C' in slot 0 (correct) → Step 2: placed 'A' in slot 2 ..."

    TODO: for step, (oi, si) in enumerate(trajectory["action_history"]):
              placed = trajectory["object_letters"][oi]
              target = trajectory["target_word"][si]
              color  = "green" if placed == target else "red"
              ax.barh(si, 1, left=step, color=color, edgecolor="white", height=0.6)
              ax.text(step + 0.5, si, placed, ha='center', va='center',
                      fontweight='bold', color='white')
    TODO: ax.set_yticks(range(WORD_LENGTH))
    TODO: ax.set_yticklabels([f"Slot {i}: {trajectory['target_word'][i]}" for i in range(WORD_LENGTH)])
    TODO: ax.set_xlabel("Step")
    TODO: ax.set_title(title)
    TODO: Add legend patches: green="Correct", red="Wrong"
    """
    for step, (oi, si) in enumerate(trajectory["action_history"]):
        placed = trajectory["object_letters"][oi]
        target = trajectory["target_word"][si]
        color  = "green" if placed == target else "red"
        ax.barh(si, 1, left=step, color=color, edgecolor="white", height=0.6)
        ax.text(step + 0.5, si, placed, ha="center", va="center",
                fontweight="bold", color="white")

    ax.set_yticks(range(WORD_LENGTH))
    ax.set_yticklabels([f"Slot {i}: {trajectory['target_word'][i]}" for i in range(WORD_LENGTH)])
    ax.set_xlabel("Step")
    ax.set_title(title)
    ax.legend(handles=[
        mpatches.Patch(color="green", label="Correct"),
        mpatches.Patch(color="red",   label="Wrong"),
    ], loc="upper right", fontsize=8)


def plot_reward_curve(ax, rl_trajectory: dict, baseline_trajectory: dict, title: str):
    """
    Overlay cumulative reward curves for the RL agent and greedy baseline.

    The gap between the two curves at the final step is the headline metric:
    how much better (or worse) the RL policy is vs. nearest-neighbour sequencing.

    TODO: steps_rl   = list(range(1, len(rl_trajectory["cumulative_rewards"]) + 1))
    TODO: steps_base = list(range(1, len(baseline_trajectory["cumulative_rewards"]) + 1))
    TODO: ax.plot(steps_rl,   rl_trajectory["cumulative_rewards"],
                  label="RL Agent",       color="steelblue", marker="o", markersize=4)
    TODO: ax.plot(steps_base, baseline_trajectory["cumulative_rewards"],
                  label="Greedy Baseline", color="darkorange", linestyle="--",
                  marker="x", markersize=4)
    TODO: ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    TODO: ax.set_xlabel("Step")
    TODO: ax.set_ylabel("Cumulative Reward")
    TODO: ax.set_title(title)
    TODO: ax.legend()
    """
    steps_rl   = list(range(1, len(rl_trajectory["cumulative_rewards"]) + 1))
    steps_base = list(range(1, len(baseline_trajectory["cumulative_rewards"]) + 1))

    ax.plot(steps_rl,   rl_trajectory["cumulative_rewards"],
            label="RL Agent",        color="steelblue",  marker="o", markersize=4)
    ax.plot(steps_base, baseline_trajectory["cumulative_rewards"],
            label="Greedy Baseline", color="darkorange", linestyle="--",
            marker="x", markersize=4)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend()


def visualise_episode(
    rl_trajectory: dict,
    baseline_trajectory: dict,
    scenario_name: str,
):
    """
    Compose a 2x2 matplotlib figure summarising one test scenario.

    Layout:
        [0,0] Workspace — RL Agent        [0,1] Workspace — Greedy Baseline
        [1,0] Cumulative Reward Curve      [1,1] Action Sequence Timeline

    The figure is saved to logs/<scenario_name>_visualisation.png
    and displayed with plt.show() (blocks until the window is closed).

    TODO: os.makedirs(LOGS_DIR, exist_ok=True)   # import LOGS_DIR from train.py
    TODO: fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    TODO: plot_workspace(axes[0, 0], rl_trajectory,       "RL Agent")
    TODO: plot_workspace(axes[0, 1], baseline_trajectory, "Greedy Baseline")
    TODO: plot_reward_curve(axes[1, 0], rl_trajectory, baseline_trajectory, "Cumulative Reward")
    TODO: plot_action_timeline(axes[1, 1], rl_trajectory,
                               f"{scenario_name} — Action Sequence (RL Agent)")
    TODO: fig.suptitle(
              f"Scenario: {scenario_name}  |  Target: {rl_trajectory['target_word']}  "
              f"|  RL: {'✓' if rl_trajectory['success'] else '✗'}  "
              f"|  Greedy: {'✓' if baseline_trajectory['success'] else '✗'}",
              fontsize=13,
          )
    TODO: plt.tight_layout()
    TODO: save_path = os.path.join(LOGS_DIR, f"{scenario_name}_visualisation.png")
    TODO: plt.savefig(save_path, dpi=150)
    TODO: print(f"Figure saved -> {save_path}")
    TODO: plt.show()
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_workspace(axes[0, 0], rl_trajectory,       "RL Agent")
    plot_workspace(axes[0, 1], baseline_trajectory, "Greedy Baseline")
    plot_reward_curve(axes[1, 0], rl_trajectory, baseline_trajectory, "Cumulative Reward")
    plot_action_timeline(
        axes[1, 1], rl_trajectory,
        f"{scenario_name} — Action Sequence (RL Agent)",
    )

    rl_tick    = "✓" if rl_trajectory["success"]       else "✗"
    base_tick  = "✓" if baseline_trajectory["success"] else "✗"
    fig.suptitle(
        f"Scenario: {scenario_name}  |  Target: {rl_trajectory['target_word']}  "
        f"|  RL: {rl_tick}  |  Greedy: {base_tick}",
        fontsize=13,
    )

    plt.tight_layout()
    save_path = os.path.join(LOGS_DIR, f"{scenario_name}_visualisation.png")
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved -> {save_path}")
    plt.close()


# ============================================================
# Main evaluation loop  (mirrors quiz2 test_policy() pattern)
# ============================================================

def test_policy():
    """
    Load the saved MaskablePPO model, run all named scenarios, and visualise results.

    For each scenario:
        1. Build WordleEnv with scenario parameters + callbacks from train.py
        2. Reset env and run the RL agent (run_episode)
        3. Reset env to the SAME initial state and run the greedy baseline (greedy_baseline)
        4. Print a one-line summary (mirrors quiz2 print pattern)
        5. Generate and save the 4-panel visualisation figure

    The RL vs. greedy delta is the primary task-level performance metric per the README.
    """
    print("Loading saved MaskablePPO model...")
    latest_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    model = MaskablePPO.load(latest_path)

    print(f"Starting evaluation across {len(SCENARIOS)} scenarios...")

    for ep, scenario in enumerate(SCENARIOS):
        name  = scenario["name"]
        stage = scenario["stage"]
        word  = scenario["target_word"]
        noise = scenario["pose_noise_std"]

        print(f"\n--- Scenario {ep + 1}: {name.upper()} ---")
        print(f"    {scenario['description']}")

        # --- Build env for this scenario ---
        env = WordleEnv(
            stage                = stage,
            target_word          = word,
            reward_callback      = custom_reward,
            observation_callback = custom_observation,
        )
        if noise != CURRICULUM_STAGES[stage - 1]["pose_noise_std"]:
            env.pose_noise_std = noise

        model.set_env(env)

        # --- RL episode ---
        rl_traj = run_episode(model, env)

        # --- Reset to same initial state for greedy baseline ---
        initial_obs, _ = env.reset(
            seed    = 42,
            options = {
                "target_word": rl_traj["target_word"],
                "poses":       rl_traj["object_poses"],
            },
        )
        base_traj = greedy_baseline(env, initial_obs)

        # --- Print summary ---
        rl_total   = rl_traj["cumulative_rewards"][-1]
        base_total = base_traj["cumulative_rewards"][-1]
        print(f"RL Agent        — Total Reward: {rl_total:.2f}  "
              f"Steps: {len(rl_traj['rewards'])}  "
              f"Success: {'✓' if rl_traj['success'] else '✗'}")
        print(f"Greedy Baseline — Total Reward: {base_total:.2f}  "
              f"Steps: {len(base_traj['rewards'])}  "
              f"Success: {'✓' if base_traj['success'] else '✗'}")
        delta = rl_total - base_total
        print(f"RL vs Greedy delta: {delta:+.2f}")

        # --- Visualise ---
        visualise_episode(rl_traj, base_traj, name)

        # Pause between scenarios — mirrors quiz2's time.sleep() pattern
        time.sleep(0.5)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    test_policy()
