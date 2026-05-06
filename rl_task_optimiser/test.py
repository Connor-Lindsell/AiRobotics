"""
Evaluation script for WordleSequencingEnv (C1–C5).

Runs the saved MaskablePPO policy and a greedy baseline across named test
scenarios, prints structured debug output per episode, and saves 2×2
matplotlib figures to logs/.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sb3_contrib import MaskablePPO

from training_env.wordle_env import (
    WordleEnv,
    ALL_POSITIONS, N_WORDLE, N_POS,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    compute_travel,
)
from train import custom_reward, MODEL_DIR, MODEL_NAME, LOGS_DIR

# ============================================================
# Test configuration
# ============================================================
RENDER_DELAY = 0.0   # seconds between steps; set >0 for live watching

# Named test scenarios — one per curriculum stage.
# target_word=None → random word per episode.
SCENARIOS = [
    {
        "name":        "c1_single_letter",
        "stage":       1,
        "target_word": None,
        "description": "C1 — 1 letter, random word, random staging slot",
        "n_episodes":  5,
    },
    {
        "name":        "c2_three_letters",
        "stage":       2,
        "target_word": None,
        "description": "C2 — 3 letters, random word",
        "n_episodes":  5,
    },
    {
        "name":        "c3_full_word",
        "stage":       3,
        "target_word": None,
        "description": "C3 — full word, 5 letters in staging",
        "n_episodes":  5,
    },
    {
        "name":        "c4_one_blocked",
        "stage":       4,
        "target_word": None,
        "description": "C4 — 1 wrong Wordle letter, must clear then fill",
        "n_episodes":  5,
    },
    {
        "name":        "c5_full_rearrange",
        "stage":       5,
        "target_word": None,
        "description": "C5 — 3–5 wrong Wordle letters, full rearrangement",
        "n_episodes":  5,
    },
]


# ============================================================
# Episode runners
# ============================================================

def _snapshot_board(env) -> dict:
    """Capture env board state for display (call after reset, before first step)."""
    return {
        "position_letter":   list(env.position_letter),
        "position_occupied": env.position_occupied.copy(),
        "wordle_correct":    env.wordle_correct.copy(),
        "target_word":       env.target_word,
        "required_slots":    set(env.required_slots),
        "robot_pos":         env.robot_pos.copy(),
    }


def run_episode(model, env) -> dict:
    """
    Run one episode with the MaskablePPO model.

    Returns:
        dict with trajectory data and debug info.
    """
    obs, _ = env.reset()
    initial_board = _snapshot_board(env)

    done               = False
    rewards            = []
    cumulative_rewards = []
    cumulative         = 0.0
    total_travel       = 0.0

    while not done:
        masks           = env.action_masks()
        action, _       = model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        total_travel += info["travel_this_step"]
        done = terminated or truncated
        if RENDER_DELAY > 0:
            time.sleep(RENDER_DELAY)

    success = info["word_complete"]

    return {
        "initial_board":     initial_board,
        "final_letter":      list(env.position_letter),
        "final_occupied":    env.position_occupied.copy(),
        "final_correct":     env.wordle_correct.copy(),
        "action_log":        list(env.action_log),
        "rewards":           rewards,
        "cumulative_rewards": cumulative_rewards,
        "total_travel":      total_travel,
        "target_word":       env.target_word,
        "success":           success,
        "n_steps":           env._step_count,
        "n_correct":         int(np.sum(env.wordle_correct)),
        "required_slots":    set(env.required_slots),
        "stage":             env.stage,
    }


def run_episode_greedy(env) -> dict:
    """
    Run one episode on an already-reset env using a greedy travel-distance baseline.

    Greedy policy: choose the valid action that minimises
        dist(robot_pos → source) + dist(source → dest)
    using the robot's current position (not always from HOME).
    """
    env.reset()
    initial_board = _snapshot_board(env)

    done               = False
    rewards            = []
    cumulative_rewards = []
    cumulative         = 0.0
    total_travel       = 0.0

    while not done:
        masks         = env.action_masks()
        valid_actions = [i for i, m in enumerate(masks) if m]
        best_action   = min(valid_actions, key=lambda a: _greedy_cost(a, env))
        _, reward, terminated, truncated, info = env.step(best_action)
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        total_travel += info["travel_this_step"]
        done = terminated or truncated

    success = info["word_complete"]

    return {
        "initial_board":      initial_board,
        "final_letter":       list(env.position_letter),
        "final_occupied":     env.position_occupied.copy(),
        "final_correct":      env.wordle_correct.copy(),
        "action_log":         list(env.action_log),
        "rewards":            rewards,
        "cumulative_rewards": cumulative_rewards,
        "total_travel":       total_travel,
        "target_word":        env.target_word,
        "success":            success,
        "n_steps":            env._step_count,
        "n_correct":          int(np.sum(env.wordle_correct)),
        "required_slots":     set(env.required_slots),
        "stage":              env.stage,
    }


def _greedy_cost(action: int, env) -> float:
    """Travel cost for a candidate action using the robot's current position."""
    source_id  = action // N_POS
    dest_id    = action %  N_POS
    source_pos = ALL_POSITIONS[source_id]
    dest_pos   = ALL_POSITIONS[dest_id]
    return compute_travel(env.robot_pos, source_pos, dest_pos)


# ============================================================
# Debug output
# ============================================================

def _board_str(position_letter, position_occupied, wordle_correct) -> str:
    wordle = " ".join(
        f"T{i}={position_letter[i] or '?'}"
        + ("✓" if wordle_correct[i] else "")
        for i in range(N_WORDLE)
    )
    staging = ", ".join(
        f"{position_letter[i]}@U{i - N_WORDLE}"
        for i in range(N_WORDLE, N_POS)
        if position_occupied[i]
    )
    return f"[{wordle}]  staging: {staging or 'empty'}"


def print_episode_debug(traj: dict, label: str) -> None:
    """Print structured debug block for one episode."""
    board = traj["initial_board"]
    stage = traj["stage"]
    word  = traj["target_word"]

    print(f"\n  [{label}]  Stage C{stage}  |  Target: {word}")
    print(f"  Init board : {_board_str(board['position_letter'], board['position_occupied'], board['wordle_correct'])}")
    print(f"  Action seq :")
    for line in traj["action_log"]:
        print(f"    {line}")
    final_board = _board_str(traj["final_letter"], traj["final_occupied"], traj["final_correct"])
    print(f"  Final board: {final_board}")
    print(
        f"  Result     : success={'✓' if traj['success'] else '✗'}"
        f"  |  steps={traj['n_steps']}"
        f"  |  travel={traj['total_travel']:.2f} m"
        f"  |  n_correct={traj['n_correct']}/{len(traj['required_slots'])}"
        f"  |  reward={traj['cumulative_rewards'][-1]:.2f}"
    )


# ============================================================
# Visualisation
# ============================================================

def plot_workspace(ax, traj: dict, title: str) -> None:
    """Draw workspace view: fixed positions, occupancy, and action arrows."""
    target_word    = traj["target_word"]
    init_board     = traj["initial_board"]
    final_letter   = traj["final_letter"]
    final_correct  = traj["final_correct"]
    action_log     = traj["action_log"]

    # Draw all named positions as faint grey dots
    for pos_id, (px, py) in enumerate(ALL_POSITIONS):
        ax.scatter(px, py, s=60, color="lightgrey", zorder=1)

    # Wordle slot boxes
    for i in range(N_WORDLE):
        sx, sy = ALL_POSITIONS[i]
        color  = "lightgreen" if final_correct[i] else ("salmon" if final_letter[i] else "lightyellow")
        box = FancyBboxPatch(
            (sx - 0.4, sy - 0.4), 0.8, 0.8,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor="black", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(sx, sy, final_letter[i] or "_", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=5)
        ax.text(sx, sy + 0.55, target_word[i], ha="center", va="bottom",
                fontsize=7, color="grey", zorder=5)

    # Staging position markers (initial occupancy)
    for pos_id in range(N_WORDLE, N_POS):
        if init_board["position_occupied"][pos_id]:
            px, py = ALL_POSITIONS[pos_id]
            ltr    = init_board["position_letter"][pos_id]
            ax.scatter(px, py, s=200, color="steelblue", zorder=3)
            ax.text(px, py, ltr or "?", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=4)

    # Action arrows — parse action_log strings to get source/dest labels
    for step_idx, log_line in enumerate(action_log):
        # Format: "Step N: move X from SRC to DST"
        parts = log_line.split()
        try:
            src_label = parts[5]   # e.g. T0 or U3
            dst_label = parts[7]   # e.g. T2 or U0
            src_id = int(src_label[1]) if src_label[0] == "T" else N_WORDLE + int(src_label[1:])
            dst_id = int(dst_label[1]) if dst_label[0] == "T" else N_WORDLE + int(dst_label[1:])
            src_pos = ALL_POSITIONS[src_id]
            dst_pos = ALL_POSITIONS[dst_id]
            rad = 0.3 if step_idx % 2 == 0 else -0.3
            arrow = FancyArrowPatch(
                src_pos, dst_pos,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>", color="darkorange", lw=1.5, zorder=3,
            )
            ax.add_patch(arrow)
            mid_x = (src_pos[0] + dst_pos[0]) / 2
            mid_y = (src_pos[1] + dst_pos[1]) / 2
            ax.text(mid_x, mid_y, str(step_idx + 1), fontsize=7, color="darkorange", zorder=6)
        except (IndexError, ValueError):
            pass   # malformed log line — skip

    ax.set_xlim(WORKSPACE_X_MIN - 0.5, WORKSPACE_X_MAX + 0.5)
    ax.set_ylim(WORKSPACE_Y_MIN - 0.5, WORKSPACE_Y_MAX + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(handles=[
        mpatches.Patch(color="lightgreen",  label="Correct Wordle slot"),
        mpatches.Patch(color="salmon",      label="Wrong Wordle slot"),
        mpatches.Patch(color="steelblue",   label="Initial staging letter"),
        mpatches.Patch(color="darkorange",  label="Action arrow"),
    ], loc="upper right", fontsize=7)


def plot_action_timeline(ax, traj: dict, title: str) -> None:
    """Horizontal bar chart: one bar per step, coloured by correct/wrong placement."""
    target_word = traj["target_word"]
    for step_idx, log_line in enumerate(traj["action_log"]):
        parts = log_line.split()
        try:
            letter    = parts[3]
            dst_label = parts[7]
            dst_id    = int(dst_label[1]) if dst_label[0] == "T" else None
            if dst_id is not None and dst_id < N_WORDLE:
                correct = (letter == target_word[dst_id])
                color   = "green" if correct else "red"
                ax.barh(dst_id, 1, left=step_idx, color=color, edgecolor="white", height=0.6)
                ax.text(step_idx + 0.5, dst_id, letter, ha="center", va="center",
                        fontweight="bold", color="white", fontsize=9)
            else:
                # staging move — show on row 5 (above Wordle rows)
                ax.barh(5, 1, left=step_idx, color="steelblue", edgecolor="white", height=0.4)
                ax.text(step_idx + 0.5, 5, letter, ha="center", va="center",
                        fontweight="bold", color="white", fontsize=8)
        except (IndexError, ValueError):
            pass

    ax.set_yticks(range(N_WORDLE + 1))
    ax.set_yticklabels(
        [f"T{i}: {target_word[i]}" for i in range(N_WORDLE)] + ["Staging"]
    )
    ax.set_xlabel("Step")
    ax.set_title(title)
    ax.legend(handles=[
        mpatches.Patch(color="green",     label="Correct Wordle placement"),
        mpatches.Patch(color="red",       label="Wrong Wordle placement"),
        mpatches.Patch(color="steelblue", label="Staging move"),
    ], loc="upper right", fontsize=8)


def plot_reward_curve(ax, rl_traj: dict, greedy_traj: dict, title: str) -> None:
    """Overlay cumulative reward curves for RL agent vs. greedy baseline."""
    steps_rl     = list(range(1, len(rl_traj["cumulative_rewards"]) + 1))
    steps_greedy = list(range(1, len(greedy_traj["cumulative_rewards"]) + 1))
    ax.plot(steps_rl,     rl_traj["cumulative_rewards"],
            label="RL Agent",        color="steelblue",  marker="o", markersize=4)
    ax.plot(steps_greedy, greedy_traj["cumulative_rewards"],
            label="Greedy Baseline", color="darkorange", linestyle="--",
            marker="x", markersize=4)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend()


def visualise_episode(rl_traj: dict, greedy_traj: dict, scenario_name: str) -> None:
    """Save a 2×2 figure: workspace (RL), workspace (greedy), reward curve, timeline."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_workspace(axes[0, 0], rl_traj,     "RL Agent")
    plot_workspace(axes[0, 1], greedy_traj, "Greedy Baseline")
    plot_reward_curve(axes[1, 0], rl_traj, greedy_traj, "Cumulative Reward")
    plot_action_timeline(axes[1, 1], rl_traj, f"{scenario_name} — RL Action Sequence")

    rl_tick = "✓" if rl_traj["success"] else "✗"
    g_tick  = "✓" if greedy_traj["success"] else "✗"
    fig.suptitle(
        f"Scenario: {scenario_name}  |  Target: {rl_traj['target_word']}"
        f"  |  RL: {rl_tick}  |  Greedy: {g_tick}",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = os.path.join(LOGS_DIR, f"{scenario_name}_visualisation.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved -> {save_path}")
    plt.close()


# ============================================================
# Aggregate metrics
# ============================================================

def print_aggregate(results: list[dict], scenario_name: str) -> None:
    """Print mean metrics across multiple episodes for one scenario."""
    n         = len(results)
    successes = sum(r["success"] for r in results)
    avg_steps = sum(r["n_steps"] for r in results) / n
    avg_trav  = sum(r["total_travel"] for r in results) / n
    avg_rew   = sum(r["cumulative_rewards"][-1] for r in results) / n
    print(
        f"\n  [{scenario_name}] n={n} | "
        f"success={successes}/{n} ({100*successes//n}%) | "
        f"avg_steps={avg_steps:.1f} | "
        f"avg_travel={avg_trav:.1f} m | "
        f"avg_reward={avg_rew:.1f}"
    )


# ============================================================
# Main evaluation loop
# ============================================================

def test_policy():
    """Load the latest model and evaluate across all C1–C5 scenarios."""
    print("Loading saved MaskablePPO model...")
    latest_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    if not os.path.exists(latest_path + ".zip"):
        print(f"No model found at {latest_path}.zip — train first.")
        return
    model = MaskablePPO.load(latest_path)

    print(f"Evaluating {len(SCENARIOS)} scenarios...\n{'='*60}")

    for scenario in SCENARIOS:
        name        = scenario["name"]
        stage       = scenario["stage"]
        target_word = scenario["target_word"]
        n_eps       = scenario["n_episodes"]
        print(f"\n{'='*60}")
        print(f"Scenario: {name.upper()}")
        print(f"  {scenario['description']}")

        env = WordleEnv(
            stage           = stage,
            reward_callback = custom_reward,
            target_word     = target_word,
        )
        model.set_env(env)

        rl_results     = []
        greedy_results = []

        for ep in range(n_eps):
            # RL episode
            rl_traj = run_episode(model, env)
            rl_results.append(rl_traj)
            print_episode_debug(rl_traj, f"RL ep{ep+1}")

            # Greedy episode (same env, fresh reset — different random seed)
            greedy_env = WordleEnv(
                stage           = stage,
                reward_callback = custom_reward,
                target_word     = target_word,
            )
            greedy_traj = run_episode_greedy(greedy_env)
            greedy_results.append(greedy_traj)
            print_episode_debug(greedy_traj, f"Greedy ep{ep+1}")

            # Visualise first episode per scenario
            if ep == 0:
                visualise_episode(rl_traj, greedy_traj, name)

        print_aggregate(rl_results,     f"RL Agent — {name}")
        print_aggregate(greedy_results, f"Greedy   — {name}")

    print(f"\n{'='*60}\nEvaluation complete.")


if __name__ == "__main__":
    test_policy()
