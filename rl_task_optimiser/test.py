"""
Evaluation script for WordleSequencingEnv (C1–C5).

Runs the saved MaskablePPO policy and a greedy baseline on *identical* tasks
(same random seed before each reset), prints per-episode head-to-head tables,
and saves per-scenario figures to logs/.

Each figure shows:
  [0,0] RL agent workspace path (episode 0)
  [0,1] Greedy workspace path  (episode 0)
  [1,0] Cumulative reward comparison (episode 0)
  [1,1] Total travel distance per episode across all N episodes (bar chart)
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sb3_contrib import MaskablePPO

from training_env.wordle_env import (
    WordleEnv,
    ALL_POSITIONS, N_CELLS,
    WORDLE_CELL_IDS, WORDLE_CELL_IDS_SET,
    FORBIDDEN_STAGING_IDS,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    compute_travel,
)
from train import custom_reward, MODEL_DIR, MODEL_NAME, LOGS_DIR

# ============================================================
# Test configuration
# ============================================================
RENDER_DELAY = 0.0

SCENARIOS = [
    {
        "name":        "c1_five_letters",
        "stage":       1,
        "target_word": None,
        "description": "C1 — 5 correct letters in staging, no distractors",
        "n_episodes":  5,
    },
    {
        "name":        "c2_with_distractors",
        "stage":       2,
        "target_word": None,
        "description": "C2 — 5 correct + 5 distractor letters, policy must discriminate",
        "n_episodes":  5,
    },
    {
        "name":        "c3_clear_and_place",
        "stage":       3,
        "target_word": None,
        "description": "C3 — all 5 Wordle slots blocked, 10 distractors, semi-constrained mask",
        "n_episodes":  5,
    },
    {
        "name":        "c4_full_autonomy",
        "stage":       4,
        "target_word": None,
        "description": "C4 — same board as C3, loose mask, full policy autonomy",
        "n_episodes":  5,
    },
]


# ============================================================
# Episode runners
# ============================================================

def _snapshot_board(env) -> dict:
    return {
        "position_letter":   list(env.position_letter),
        "position_occupied": env.position_occupied.copy(),
        "wordle_correct":    env.wordle_correct.copy(),
        "target_word":       env.target_word,
        "required_slots":    set(env.required_slots),
        "robot_pos":         env.robot_pos.copy(),
    }


def run_episode(model, env) -> dict:
    """Run one RL episode. Caller must seed random before calling."""
    obs, _ = env.reset()
    initial_board = _snapshot_board(env)

    done = False
    rewards, cumulative_rewards = [], []
    cumulative       = 0.0
    total_travel     = 0.0
    all_path_segments = []

    while not done:
        masks          = env.action_masks()
        action, _      = model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        total_travel += info["travel_this_step"]
        all_path_segments.append(info.get("path_segments", []))
        done = terminated or truncated
        if RENDER_DELAY > 0:
            time.sleep(RENDER_DELAY)

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
        "success":            info["word_complete"],
        "n_steps":            env._step_count,
        "n_correct":          int(np.sum(env.wordle_correct)),
        "required_slots":     set(env.required_slots),
        "stage":              env.stage,
        "path_segments":      all_path_segments,
    }


def run_episode_greedy(env) -> dict:
    """Run one greedy episode. Caller must seed random before calling."""
    env.reset()
    initial_board = _snapshot_board(env)

    done = False
    rewards, cumulative_rewards = [], []
    cumulative        = 0.0
    total_travel      = 0.0
    all_path_segments = []

    while not done:
        masks         = env.action_masks()
        valid_actions = [i for i, m in enumerate(masks) if m]
        best_action   = min(valid_actions, key=lambda a: _greedy_cost(a, env))
        _, reward, terminated, truncated, info = env.step(best_action)
        rewards.append(reward)
        cumulative += reward
        cumulative_rewards.append(cumulative)
        total_travel += info["travel_this_step"]
        all_path_segments.append(info.get("path_segments", []))
        done = terminated or truncated

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
        "success":            info["word_complete"],
        "n_steps":            env._step_count,
        "n_correct":          int(np.sum(env.wordle_correct)),
        "required_slots":     set(env.required_slots),
        "stage":              env.stage,
        "path_segments":      all_path_segments,
    }


def _greedy_cost(action: int, env) -> float:
    source_id  = action // N_CELLS
    dest_id    = action %  N_CELLS
    return compute_travel(env.robot_pos, ALL_POSITIONS[source_id], ALL_POSITIONS[dest_id])


# ============================================================
# Debug output
# ============================================================

def _board_str(position_letter, position_occupied, wordle_correct) -> str:
    wordle = " ".join(
        f"W{wi}={position_letter[cid] or '?'}"
        + ("✓" if wordle_correct[wi] else "")
        for wi, cid in enumerate(WORDLE_CELL_IDS)
    )
    staging = ", ".join(
        f"{position_letter[i]}@G{i}"
        for i in range(N_CELLS)
        if i not in WORDLE_CELL_IDS_SET and position_occupied[i]
    )
    return f"[{wordle}]  staging: {staging or 'empty'}"


def print_episode_debug(traj: dict, label: str) -> None:
    board = traj["initial_board"]
    print(f"\n  [{label}]  Stage C{traj['stage']}  |  Target: {traj['target_word']}")
    print(f"  Init board : {_board_str(board['position_letter'], board['position_occupied'], board['wordle_correct'])}")
    print(f"  Action seq :")
    for line in traj["action_log"]:
        print(f"    {line}")
    print(f"  Final board: {_board_str(traj['final_letter'], traj['final_occupied'], traj['final_correct'])}")
    print(
        f"  Result     : success={'✓' if traj['success'] else '✗'}"
        f"  |  steps={traj['n_steps']}"
        f"  |  travel={traj['total_travel']:.2f} m"
        f"  |  n_correct={traj['n_correct']}/{len(traj['required_slots'])}"
        f"  |  reward={traj['cumulative_rewards'][-1]:.2f}"
    )


def print_head_to_head(rl_traj: dict, greedy_traj: dict, ep_num: int) -> None:
    """Print a side-by-side comparison table for a single episode pair."""
    rl_rew     = rl_traj["cumulative_rewards"][-1]
    g_rew      = greedy_traj["cumulative_rewards"][-1]
    rl_suc     = "✓" if rl_traj["success"] else "✗"
    g_suc      = "✓" if greedy_traj["success"] else "✗"

    print(f"\n  ┌─ Head-to-Head ep{ep_num}  (target: {rl_traj['target_word']}) ──────────────────┐")
    print(f"  │  {'Metric':<22} {'RL Agent':>10} {'Greedy':>10} {'Delta (RL−G)':>13} │")
    print(f"  │  {'─'*57} │")
    _row("Steps",        rl_traj["n_steps"],     greedy_traj["n_steps"],     fmt="d")
    _row("Travel (m)",   rl_traj["total_travel"], greedy_traj["total_travel"], fmt=".2f")
    _row("Reward",       rl_rew,                  g_rew,                       fmt=".2f")
    print(f"  │  {'Success':<22} {rl_suc:>10} {g_suc:>10} {'':>13} │")
    print(f"  └{'─'*59}┘")


def _row(label: str, rl_val, g_val, fmt: str = ".2f") -> None:
    delta = rl_val - g_val
    sign  = "+" if delta > 0 else ""
    print(
        f"  │  {label:<22} {rl_val:>10{fmt}} {g_val:>10{fmt}} "
        f"{sign}{delta:>12{fmt}} │"
    )


def print_aggregate_comparison(rl_results: list[dict], greedy_results: list[dict], scenario_name: str) -> None:
    n = len(rl_results)

    def _avg(results, key): return sum(r[key] for r in results) / n
    def _avg_rew(results):  return sum(r["cumulative_rewards"][-1] for r in results) / n
    def _suc(results):      return sum(r["success"] for r in results)

    rl_steps  = _avg(rl_results, "n_steps");       g_steps  = _avg(greedy_results, "n_steps")
    rl_travel = _avg(rl_results, "total_travel");  g_travel = _avg(greedy_results, "total_travel")
    rl_rew    = _avg_rew(rl_results);              g_rew    = _avg_rew(greedy_results)
    rl_suc    = _suc(rl_results);                  g_suc    = _suc(greedy_results)

    print(f"\n  ╔═ Aggregate [{scenario_name}]  n={n} ══════════════════════════════╗")
    print(f"  ║  {'Metric':<22} {'RL Agent':>10} {'Greedy':>10} {'Delta (RL−G)':>13} ║")
    print(f"  ║  {'═'*57} ║")
    _row("Avg steps",    rl_steps,  g_steps,  fmt=".1f")
    _row("Avg travel (m)", rl_travel, g_travel, fmt=".2f")
    _row("Avg reward",   rl_rew,    g_rew,    fmt=".2f")
    print(f"  ║  {'Success rate':<22} {f'{rl_suc}/{n}':>10} {f'{g_suc}/{n}':>10} {'':>13} ║")
    print(f"  ╚{'═'*59}╝")


# ============================================================
# Visualisation
# ============================================================

def plot_workspace(ax, traj: dict, title: str) -> None:
    target_word   = traj["target_word"]
    init_board    = traj["initial_board"]
    final_letter  = traj["final_letter"]
    final_correct = traj["final_correct"]

    for cell_id, (px, py) in enumerate(ALL_POSITIONS):
        color = "#FFB3B3" if cell_id in FORBIDDEN_STAGING_IDS else "lightgrey"
        ax.scatter(px, py, s=20, color=color, zorder=1)

    for wi, cid in enumerate(WORDLE_CELL_IDS):
        sx, sy = ALL_POSITIONS[cid]
        ltr    = final_letter[cid]
        color  = "lightgreen" if final_correct[wi] else ("salmon" if ltr else "lightyellow")
        ax.add_patch(FancyBboxPatch(
            (sx - 0.3, sy - 0.3), 0.6, 0.6,
            boxstyle="round,pad=0.04", linewidth=1.5,
            edgecolor="black", facecolor=color, zorder=2,
        ))
        ax.text(sx, sy, ltr or "_", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=5)
        ax.text(sx, sy + 0.38, target_word[wi], ha="center", va="bottom",
                fontsize=6, color="grey", zorder=5)

    for cell_id in range(N_CELLS):
        if cell_id not in WORDLE_CELL_IDS_SET and init_board["position_occupied"][cell_id]:
            px, py = ALL_POSITIONS[cell_id]
            ltr    = init_board["position_letter"][cell_id]
            ax.scatter(px, py, s=160, color="steelblue", zorder=3)
            ax.text(px, py, ltr or "?", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=4)

    seg_colors = ["purple", "darkorange"]
    for step_idx, segments in enumerate(traj.get("path_segments", [])):
        rad = 0.3 if step_idx % 2 == 0 else -0.3
        for seg_idx, (from_pos, to_pos) in enumerate(segments):
            ax.add_patch(FancyArrowPatch(
                from_pos, to_pos,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color=seg_colors[seg_idx],
                lw=1.2, zorder=3,
            ))
        if len(segments) == 2:
            _, s = segments[0]
            _, d = segments[1]
            ax.text((s[0] + d[0]) / 2, (s[1] + d[1]) / 2,
                    str(step_idx + 1), fontsize=6, color="darkorange", zorder=6)

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
        mpatches.Patch(color="purple",      label="Robot travel to source"),
        mpatches.Patch(color="darkorange",  label="Source to destination"),
        mpatches.Patch(color="#FFB3B3",     label="Forbidden zone (C4/C5)"),
    ], loc="upper right", fontsize=6)


def plot_reward_curve(ax, rl_traj: dict, greedy_traj: dict, title: str) -> None:
    steps_rl     = list(range(1, len(rl_traj["cumulative_rewards"]) + 1))
    steps_greedy = list(range(1, len(greedy_traj["cumulative_rewards"]) + 1))
    ax.plot(steps_rl,     rl_traj["cumulative_rewards"],
            label="RL Agent",        color="steelblue", marker="o", markersize=4)
    ax.plot(steps_greedy, greedy_traj["cumulative_rewards"],
            label="Greedy Baseline", color="darkorange", linestyle="--",
            marker="x", markersize=4)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend()


def plot_travel_comparison(ax, rl_results: list[dict], greedy_results: list[dict], title: str) -> None:
    """Bar chart of total travel distance per episode, RL vs Greedy side-by-side."""
    n        = len(rl_results)
    episodes = list(range(1, n + 1))
    rl_vals  = [r["total_travel"] for r in rl_results]
    g_vals   = [r["total_travel"] for r in greedy_results]

    x     = np.arange(n)
    width = 0.35

    bars_rl = ax.bar(x - width / 2, rl_vals, width, label="RL Agent",
                     color="steelblue", alpha=0.85)
    bars_g  = ax.bar(x + width / 2, g_vals,  width, label="Greedy Baseline",
                     color="darkorange", alpha=0.85)

    # Label each bar with its value
    for bar in bars_rl:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    # Avg lines
    ax.axhline(np.mean(rl_vals), color="steelblue", linestyle=":",
               linewidth=1.2, label=f"RL avg {np.mean(rl_vals):.1f} m")
    ax.axhline(np.mean(g_vals), color="darkorange", linestyle=":",
               linewidth=1.2, label=f"Greedy avg {np.mean(g_vals):.1f} m")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ep{i}" for i in episodes])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Travel Distance (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)


def visualise_scenario(rl_results: list[dict], greedy_results: list[dict], scenario_name: str) -> None:
    """
    Save a 2×2 figure for the scenario:
      [0,0] RL workspace path (episode 0)
      [0,1] Greedy workspace path (episode 0)
      [1,0] Cumulative reward comparison (episode 0)
      [1,1] Travel distance per episode — all N episodes, RL vs Greedy
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rl0     = rl_results[0]
    greedy0 = greedy_results[0]

    plot_workspace(axes[0, 0], rl0,     f"RL Agent  (ep1, target: {rl0['target_word']})")
    plot_workspace(axes[0, 1], greedy0, f"Greedy Baseline  (ep1, target: {greedy0['target_word']})")
    plot_reward_curve(axes[1, 0], rl0, greedy0, "Cumulative Reward — ep1")
    plot_travel_comparison(axes[1, 1], rl_results, greedy_results,
                           f"Total Travel Distance — all {len(rl_results)} episodes")

    rl_tick = "✓" if rl0["success"] else "✗"
    g_tick  = "✓" if greedy0["success"] else "✗"
    fig.suptitle(
        f"Scenario: {scenario_name}  |  RL ep1: {rl_tick}  |  Greedy ep1: {g_tick}",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = os.path.join(LOGS_DIR, f"{scenario_name}_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Figure saved -> {save_path}")
    plt.close()


# ============================================================
# Main evaluation loop
# ============================================================

def test_policy():
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

        greedy_env = WordleEnv(
            stage           = stage,
            reward_callback = custom_reward,
            target_word     = target_word,
        )

        rl_results     = []
        greedy_results = []

        for ep in range(n_eps):
            # Seed Python's random module before both resets so both agents
            # face identical initial boards (same target word, same staging layout).
            episode_seed = random.randint(0, 2**31 - 1)

            random.seed(episode_seed)
            rl_traj = run_episode(model, env)

            random.seed(episode_seed)
            greedy_traj = run_episode_greedy(greedy_env)

            rl_results.append(rl_traj)
            greedy_results.append(greedy_traj)

            print_episode_debug(rl_traj,     f"RL ep{ep+1}")
            print_episode_debug(greedy_traj, f"Greedy ep{ep+1}")
            print_head_to_head(rl_traj, greedy_traj, ep + 1)

        visualise_scenario(rl_results, greedy_results, name)
        print_aggregate_comparison(rl_results, greedy_results, name)

    print(f"\n{'='*60}\nEvaluation complete.")


if __name__ == "__main__":
    test_policy()
