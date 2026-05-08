"""
Evaluation script for WordleSequencingEnv (C1–C4).

Runs the saved MaskablePPO policy and a greedy baseline on *identical* tasks
(same random seed before each reset), prints per-episode head-to-head tables,
and saves per-scenario figures to logs/.

Each figure shows:
  [0,0] RL agent workspace path (best episode)
  [0,1] Greedy workspace path  (best episode)
  [1,0] Cumulative reward comparison (best episode)
  [1,1] Total travel distance per episode across all N episodes (bar chart)

ROS2 integration
----------------
The `TaskSequencerEvaluator` class is designed to be subclassed by a ROS2 node:

    class TaskSequencerNode(TaskSequencerEvaluator, Node):
        def word_callback(self, msg):
            env = self.build_env(stage=3, word=msg.data,
                                 fixed_positions=self._block_positions)
            traj = self.run_episode(env)
            sequence = self.get_task_sequence(traj)
            self.publish_sequence(sequence)

        def blocks_callback(self, msg):
            # msg contains list of {letter, x_m, y_m} from perception
            self._block_positions = self._snap_to_grid(msg)

        def _snap_to_grid(self, msg) -> dict[int, str]:
            # Snap physical (x,y) to nearest cell_id via ALL_POSITIONS
            ...

The three ROS2-facing methods are:
  build_env(stage, word, fixed_positions)  — create env from ROS inputs
  run_episode(env)                         — run policy, return trajectory
  get_task_sequence(trajectory)            — convert trajectory to task list
"""

import argparse
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from sb3_contrib import MaskablePPO

from training_env.wordle_env import (
    WordleEnv,
    HOLDOUT_WORD,
    ALL_POSITIONS, N_CELLS,
    WORDLE_CELL_IDS, WORDLE_CELL_IDS_SET, WORDLE_CELL_ID_TO_IDX,
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

# ============================================================
# Animation configuration
# ============================================================
ANIMATE         = True   # set False to skip GIF generation
ANIM_FPS        = 20     # frames per second in saved GIF
ANIM_SPEED_MPS  = 1.5    # metres per second — end-effector constant travel speed
ANIM_HOLD       = 8      # frames to hold on the final board state after each step

SCENARIOS = [
    {
        "name":        "c1_five_letters",
        "stage":       1,
        "target_word": HOLDOUT_WORD,
        "description": "C1 — 5 correct letters in staging, no distractors",
        "n_episodes":  5,
    },
    {
        "name":        "c2_with_distractors",
        "stage":       2,
        "target_word": HOLDOUT_WORD,
        "description": "C2 — 5 correct + 5 distractor letters, policy must discriminate",
        "n_episodes":  5,
    },
    {
        "name":        "c3_clear_and_place",
        "stage":       3,
        "target_word": HOLDOUT_WORD,
        "description": "C3 — all 5 Wordle slots blocked, 10 distractors, semi-constrained mask",
        "n_episodes":  5,
    },
    {
        "name":        "c4_full_autonomy",
        "stage":       4,
        "target_word": HOLDOUT_WORD,
        "description": "C4 — same board as C3, loose mask, full policy autonomy",
        "n_episodes":  5,
    },
]


# ============================================================
# Main evaluator class
# ============================================================

class TaskSequencerEvaluator:
    """
    Loads a trained MaskablePPO policy and evaluates it against a greedy baseline.

    Designed for both standalone test runs and subclassing by a ROS2 node.
    The three ROS2-facing methods are build_env, run_episode, and get_task_sequence.
    """

    def __init__(self, model_path: str, reward_callback):
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"No model at {model_path}.zip — train first.")
        self.model = MaskablePPO.load(model_path)
        self.reward_callback = reward_callback

    # ----------------------------------------------------------
    # ROS2-facing interface
    # ----------------------------------------------------------

    def build_env(
        self,
        stage: int,
        word: str | None = None,
        fixed_positions: dict[int, str] | None = None,
    ) -> WordleEnv:
        """
        Create a WordleEnv for inference.

        stage           — curriculum stage (1–4)
        word            — target word; None draws randomly from WORD_LIST
        fixed_positions — {cell_id: letter} from a perception system;
                          when set, reset() uses these positions instead of
                          random staging placement (ROS2 hook)
        """
        return WordleEnv(
            stage=stage,
            reward_callback=self.reward_callback,
            target_word=word,
            fixed_initial_positions=fixed_positions,
        )

    def run_episode(self, env: WordleEnv, model: MaskablePPO | None = None) -> dict:
        """
        Run one RL episode. Caller must seed random before calling if reproducibility
        is needed. Returns a trajectory dict.
        """
        m = model or self.model
        m.set_env(env)
        obs, _ = env.reset()
        initial_board = self._snapshot_board(env)

        done = False
        rewards, cumulative_rewards = [], []
        cumulative        = 0.0
        total_travel      = 0.0
        all_path_segments = []
        step_frames       = []

        while not done:
            masks          = env.action_masks()
            action, _      = m.predict(obs, deterministic=True, action_masks=masks)
            action         = int(action)
            source_id      = action // N_CELLS
            board_before   = self._snapshot_board(env)
            letter_moved   = env.position_letter[source_id]

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            cumulative += reward
            cumulative_rewards.append(cumulative)
            total_travel += info["travel_this_step"]
            all_path_segments.append(info.get("path_segments", []))
            step_frames.append({
                "source_id":    source_id,
                "dest_id":      action % N_CELLS,
                "robot_from":   info["robot_pos_before"],
                "source_pos":   info["source_pos"],
                "dest_pos":     info["dest_pos"],
                "letter":       letter_moved,
                "board_before": board_before,
                "board_after":  self._snapshot_board(env),
            })
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
            "step_frames":        step_frames,
        }

    def get_task_sequence(self, trajectory: dict) -> list[dict]:
        """
        Convert a trajectory into a ROS2-ready task sequence.

        Each entry describes one robot action:
          step          — 1-based action index
          description   — human-readable summary
          pick_pose     — (x_m, y_m) world coordinates of source cell centre
          place_pose    — (x_m, y_m) world coordinates of destination cell centre
          source_cell_id / dest_cell_id — grid indices (0–90)
          letter        — letter being moved

        The ROS2 node adds a fixed z offset when building geometry_msgs/Pose.
        """
        sequence = []
        for i, frame in enumerate(trajectory["step_frames"], 1):
            src_id = frame["source_id"]
            dst_id = frame["dest_id"]
            letter = frame["letter"] or "?"
            if dst_id in WORDLE_CELL_IDS_SET:
                slot_idx = WORDLE_CELL_ID_TO_IDX[dst_id]
                desc = f"Pick '{letter}' from cell {src_id} and place in Wordle slot {slot_idx}"
            else:
                desc = f"Move '{letter}' from cell {src_id} to staging cell {dst_id}"
            sequence.append({
                "step":           i,
                "description":    desc,
                "pick_pose":      ALL_POSITIONS[src_id],
                "place_pose":     ALL_POSITIONS[dst_id],
                "source_cell_id": src_id,
                "dest_cell_id":   dst_id,
                "letter":         letter,
            })
        return sequence

    # ----------------------------------------------------------
    # Multi-episode evaluation helpers
    # ----------------------------------------------------------

    def run_episode_greedy(self, env: WordleEnv) -> dict:
        """Run one greedy episode. Caller must seed random before calling."""
        env.reset()
        initial_board = self._snapshot_board(env)

        done = False
        rewards, cumulative_rewards = [], []
        cumulative        = 0.0
        total_travel      = 0.0
        all_path_segments = []
        step_frames       = []

        while not done:
            masks         = env.action_masks()
            valid_actions = [i for i, m in enumerate(masks) if m]
            best_action   = min(valid_actions, key=lambda a: self._greedy_cost(a, env))
            source_id     = best_action // N_CELLS
            board_before  = self._snapshot_board(env)
            letter_moved  = env.position_letter[source_id]

            _, reward, terminated, truncated, info = env.step(best_action)
            rewards.append(reward)
            cumulative += reward
            cumulative_rewards.append(cumulative)
            total_travel += info["travel_this_step"]
            all_path_segments.append(info.get("path_segments", []))
            step_frames.append({
                "source_id":    source_id,
                "dest_id":      best_action % N_CELLS,
                "robot_from":   info["robot_pos_before"],
                "source_pos":   info["source_pos"],
                "dest_pos":     info["dest_pos"],
                "letter":       letter_moved,
                "board_before": board_before,
                "board_after":  self._snapshot_board(env),
            })
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
            "step_frames":        step_frames,
        }

    def run_best_of_n(
        self, stage: int, word: str, n: int = 5
    ) -> tuple[int, list[dict], list[dict]]:
        """
        Run N seeded episode pairs (RL + greedy) for a given stage/word.

        Returns (best_idx, rl_results, greedy_results) where best_idx is the
        index of the RL episode with the lowest travel among successes (fallback:
        lowest travel overall).
        """
        env        = self.build_env(stage=stage, word=word)
        greedy_env = self.build_env(stage=stage, word=word)

        rl_results     = []
        greedy_results = []

        for _ in range(n):
            episode_seed = random.randint(0, 2**31 - 1)
            random.seed(episode_seed)
            rl_results.append(self.run_episode(env))
            random.seed(episode_seed)
            greedy_results.append(self.run_episode_greedy(greedy_env))

        successful_idx = [i for i, t in enumerate(rl_results) if t["success"]]
        if successful_idx:
            best_idx = min(successful_idx, key=lambda i: rl_results[i]["total_travel"])
        else:
            best_idx = min(range(n), key=lambda i: rl_results[i]["total_travel"])

        return best_idx, rl_results, greedy_results

    def run_evaluation(self, scenarios: list[dict], animate: bool = ANIMATE) -> None:
        """
        Full evaluation loop — runs all scenarios, prints tables, saves figures.
        Animates the best (lowest travel among successes) episode per scenario
        unless animate=False.
        """
        print(f"Evaluating {len(scenarios)} scenarios...\n{'='*60}")

        for scenario in scenarios:
            name        = scenario["name"]
            stage       = scenario["stage"]
            target_word = scenario.get("target_word")
            n_eps       = scenario["n_episodes"]

            print(f"\n{'='*60}")
            print(f"Scenario: {name.upper()}  (target: {target_word or 'random'})")
            print(f"  {scenario['description']}")

            env        = self.build_env(stage=stage, word=target_word)
            greedy_env = self.build_env(stage=stage, word=target_word)
            self.model.set_env(env)

            rl_results     = []
            greedy_results = []

            for ep in range(n_eps):
                episode_seed = random.randint(0, 2**31 - 1)

                random.seed(episode_seed)
                rl_traj = self.run_episode(env)

                random.seed(episode_seed)
                greedy_traj = self.run_episode_greedy(greedy_env)

                rl_results.append(rl_traj)
                greedy_results.append(greedy_traj)

                self.print_episode_debug(rl_traj,     f"RL ep{ep+1}")
                self.print_episode_debug(greedy_traj, f"Greedy ep{ep+1}")
                self.print_head_to_head(rl_traj, greedy_traj, ep + 1)

            # Find best RL episode to animate
            successful_idx = [i for i, t in enumerate(rl_results) if t["success"]]
            if successful_idx:
                best_idx = min(successful_idx, key=lambda i: rl_results[i]["total_travel"])
            else:
                best_idx = min(range(n_eps), key=lambda i: rl_results[i]["total_travel"])

            self.visualise_scenario(rl_results, greedy_results, name, best_idx)

            if animate:
                self.animate_episode(
                    rl_results[best_idx], greedy_results[best_idx],
                    name, ep=best_idx,
                )

            self.print_aggregate_comparison(rl_results, greedy_results, name)

        print(f"\n{'='*60}\nEvaluation complete.")

    # ----------------------------------------------------------
    # Debug output
    # ----------------------------------------------------------

    def print_episode_debug(self, traj: dict, label: str) -> None:
        board = traj["initial_board"]
        print(f"\n  [{label}]  Stage C{traj['stage']}  |  Target: {traj['target_word']}")
        print(f"  Init board : {self._board_str(board['position_letter'], board['position_occupied'], board['wordle_correct'])}")
        print(f"  Action seq :")
        for line in traj["action_log"]:
            print(f"    {line}")
        print(f"  Final board: {self._board_str(traj['final_letter'], traj['final_occupied'], traj['final_correct'])}")
        print(
            f"  Result     : success={'✓' if traj['success'] else '✗'}"
            f"  |  steps={traj['n_steps']}"
            f"  |  travel={traj['total_travel']:.2f} m"
            f"  |  n_correct={traj['n_correct']}/{len(traj['required_slots'])}"
            f"  |  reward={traj['cumulative_rewards'][-1]:.2f}"
        )

    def print_head_to_head(self, rl_traj: dict, greedy_traj: dict, ep_num: int) -> None:
        rl_rew = rl_traj["cumulative_rewards"][-1]
        g_rew  = greedy_traj["cumulative_rewards"][-1]
        rl_suc = "✓" if rl_traj["success"] else "✗"
        g_suc  = "✓" if greedy_traj["success"] else "✗"

        print(f"\n  ┌─ Head-to-Head ep{ep_num}  (target: {rl_traj['target_word']}) ──────────────────┐")
        print(f"  │  {'Metric':<22} {'RL Agent':>10} {'Greedy':>10} {'Delta (RL−G)':>13} │")
        print(f"  │  {'─'*57} │")
        self._row("Steps",        rl_traj["n_steps"],      greedy_traj["n_steps"],      fmt="d")
        self._row("Travel (m)",   rl_traj["total_travel"],  greedy_traj["total_travel"],  fmt=".2f")
        self._row("Reward",       rl_rew,                   g_rew,                        fmt=".2f")
        print(f"  │  {'Success':<22} {rl_suc:>10} {g_suc:>10} {'':>13} │")
        print(f"  └{'─'*59}┘")

    def print_aggregate_comparison(
        self, rl_results: list[dict], greedy_results: list[dict], scenario_name: str
    ) -> None:
        n = len(rl_results)

        def _avg(results, key):    return sum(r[key] for r in results) / n
        def _avg_rew(results):     return sum(r["cumulative_rewards"][-1] for r in results) / n
        def _suc(results):         return sum(r["success"] for r in results)

        rl_steps  = _avg(rl_results, "n_steps");       g_steps  = _avg(greedy_results, "n_steps")
        rl_travel = _avg(rl_results, "total_travel");  g_travel = _avg(greedy_results, "total_travel")
        rl_rew    = _avg_rew(rl_results);              g_rew    = _avg_rew(greedy_results)
        rl_suc    = _suc(rl_results);                  g_suc    = _suc(greedy_results)

        print(f"\n  ╔═ Aggregate [{scenario_name}]  n={n} ══════════════════════════════╗")
        print(f"  ║  {'Metric':<22} {'RL Agent':>10} {'Greedy':>10} {'Delta (RL−G)':>13} ║")
        print(f"  ║  {'═'*57} ║")
        self._row("Avg steps",      rl_steps,  g_steps,  fmt=".1f", box="║")
        self._row("Avg travel (m)", rl_travel, g_travel, fmt=".2f", box="║")
        self._row("Avg reward",     rl_rew,    g_rew,    fmt=".2f", box="║")
        print(f"  ║  {'Success rate':<22} {f'{rl_suc}/{n}':>10} {f'{g_suc}/{n}':>10} {'':>13} ║")
        print(f"  ╚{'═'*59}╝")

    # ----------------------------------------------------------
    # Visualisation
    # ----------------------------------------------------------

    def visualise_scenario(
        self,
        rl_results: list[dict],
        greedy_results: list[dict],
        scenario_name: str,
        best_idx: int = 0,
    ) -> None:
        """
        Save a 2×2 figure for the scenario using the best episode for workspace/reward plots.
        """
        os.makedirs(LOGS_DIR, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        rl_best     = rl_results[best_idx]
        greedy_best = greedy_results[best_idx]

        ep_label = f"ep{best_idx+1} (best)"
        self.plot_workspace(axes[0, 0], rl_best,     f"RL Agent  ({ep_label}, target: {rl_best['target_word']})")
        self.plot_workspace(axes[0, 1], greedy_best, f"Greedy Baseline  ({ep_label}, target: {greedy_best['target_word']})")
        self.plot_reward_curve(axes[1, 0], rl_best, greedy_best, f"Cumulative Reward — {ep_label}")
        self.plot_travel_comparison(axes[1, 1], rl_results, greedy_results,
                                    f"Total Travel Distance — all {len(rl_results)} episodes")

        rl_tick = "✓" if rl_best["success"] else "✗"
        g_tick  = "✓" if greedy_best["success"] else "✗"
        fig.suptitle(
            f"Scenario: {scenario_name}  |  RL {ep_label}: {rl_tick}  |  Greedy {ep_label}: {g_tick}",
            fontsize=13,
        )
        plt.tight_layout()
        save_path = os.path.join(LOGS_DIR, f"{scenario_name}_comparison.png")
        plt.savefig(save_path, dpi=150)
        print(f"  Figure saved -> {save_path}")
        plt.close()

    def plot_workspace(self, ax, traj: dict, title: str) -> None:
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

    def plot_reward_curve(self, ax, rl_traj: dict, greedy_traj: dict, title: str) -> None:
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

    def plot_travel_comparison(
        self,
        ax,
        rl_results: list[dict],
        greedy_results: list[dict],
        title: str,
    ) -> None:
        n        = len(rl_results)
        rl_vals  = [r["total_travel"] for r in rl_results]
        g_vals   = [r["total_travel"] for r in greedy_results]

        x     = np.arange(n)
        width = 0.35

        bars_rl = ax.bar(x - width / 2, rl_vals, width, label="RL Agent",
                         color="steelblue", alpha=0.85)
        bars_g  = ax.bar(x + width / 2, g_vals,  width, label="Greedy Baseline",
                         color="darkorange", alpha=0.85)

        for bar in bars_rl:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
        for bar in bars_g:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(np.mean(rl_vals), color="steelblue", linestyle=":",
                   linewidth=1.2, label=f"RL avg {np.mean(rl_vals):.1f} m")
        ax.axhline(np.mean(g_vals), color="darkorange", linestyle=":",
                   linewidth=1.2, label=f"Greedy avg {np.mean(g_vals):.1f} m")

        ax.set_xticks(x)
        ax.set_xticklabels([f"ep{i}" for i in range(1, n + 1)])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Travel Distance (m)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # ----------------------------------------------------------
    # Animation
    # ----------------------------------------------------------

    def animate_episode(
        self,
        rl_traj: dict,
        greedy_traj: dict,
        scenario_name: str,
        ep: int = 0,
    ) -> None:
        """
        Build and save a side-by-side GIF of RL vs Greedy for one episode.
        The shorter trajectory is padded with its last frame so both finish
        at the same time.
        """
        rl_frames     = self._build_animation_frames(rl_traj)
        greedy_frames = self._build_animation_frames(greedy_traj)

        max_len = max(len(rl_frames), len(greedy_frames))
        rl_frames     += [rl_frames[-1]]     * (max_len - len(rl_frames))
        greedy_frames += [greedy_frames[-1]] * (max_len - len(greedy_frames))

        target = rl_traj["target_word"]
        rl_suc = "✓" if rl_traj["success"] else "✗"
        g_suc  = "✓" if greedy_traj["success"] else "✗"

        fig, (ax_rl, ax_g) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f"{scenario_name}  |  target: {target}"
            f"  |  RL {rl_suc} ({rl_traj['n_steps']} steps,"
            f" {rl_traj['total_travel']:.2f} m)"
            f"  |  Greedy {g_suc} ({greedy_traj['n_steps']} steps,"
            f" {greedy_traj['total_travel']:.2f} m)",
            fontsize=9,
        )

        def _update(i):
            self._draw_anim_frame(ax_rl, rl_frames[i],     target, f"RL Agent (ep{ep+1})")
            self._draw_anim_frame(ax_g,  greedy_frames[i], target, f"Greedy Baseline (ep{ep+1})")
            return []

        anim = FuncAnimation(fig, _update, frames=max_len,
                             interval=1000 / ANIM_FPS, blit=False)

        os.makedirs(LOGS_DIR, exist_ok=True)
        save_path = os.path.join(LOGS_DIR, f"{scenario_name}_ep{ep+1}_animation.gif")
        print(f"  Saving animation ({max_len} frames @ {ANIM_FPS} fps) -> {save_path}")
        anim.save(save_path, writer=PillowWriter(fps=ANIM_FPS))
        plt.close()
        print(f"  Animation saved -> {save_path}")

    # ----------------------------------------------------------
    # Static / private helpers
    # ----------------------------------------------------------

    @staticmethod
    def _snapshot_board(env) -> dict:
        return {
            "position_letter":   list(env.position_letter),
            "position_occupied": env.position_occupied.copy(),
            "wordle_correct":    env.wordle_correct.copy(),
            "target_word":       env.target_word,
            "required_slots":    set(env.required_slots),
            "robot_pos":         env.robot_pos.copy(),
        }

    @staticmethod
    def _greedy_cost(action: int, env) -> float:
        source_id = action // N_CELLS
        dest_id   = action %  N_CELLS
        return compute_travel(env.robot_pos, ALL_POSITIONS[source_id], ALL_POSITIONS[dest_id])

    @staticmethod
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

    @staticmethod
    def _row(label: str, rl_val, g_val, fmt: str = ".2f", box: str = "│") -> None:
        delta = rl_val - g_val
        sign  = "+" if delta > 0 else ""
        print(
            f"  {box}  {label:<22} {rl_val:>10{fmt}} {g_val:>10{fmt}} "
            f"{sign}{delta:>12{fmt}} {box}"
        )

    @staticmethod
    def _build_animation_frames(traj: dict) -> list[dict]:
        frames = []
        for step_num, step in enumerate(traj["step_frames"], start=1):
            rf     = np.array(step["robot_from"], dtype=float)
            sp     = np.array(step["source_pos"],  dtype=float)
            dp     = np.array(step["dest_pos"],    dtype=float)
            src_id = step["source_id"]
            letter = step["letter"]
            bb     = step["board_before"]
            ba     = step["board_after"]

            board_transit = {
                "position_letter":   list(bb["position_letter"]),
                "position_occupied": bb["position_occupied"].copy(),
                "wordle_correct":    bb["wordle_correct"].copy(),
                "target_word":       bb["target_word"],
                "required_slots":    bb["required_slots"],
                "robot_pos":         bb["robot_pos"],
            }
            board_transit["position_letter"][src_id]   = None
            board_transit["position_occupied"][src_id] = False

            base = {"step_num": step_num, "total_steps": len(traj["step_frames"])}

            def _seg_frames(a, b):
                dist = float(np.linalg.norm(b - a))
                return max(1, round(dist / ANIM_SPEED_MPS * ANIM_FPS))

            for t in np.linspace(0, 1, _seg_frames(rf, sp), endpoint=False):
                pos = rf + t * (sp - rf)
                frames.append({**base, "robot_xy": tuple(pos),
                               "carrying": None, "board": bb})

            for t in np.linspace(0, 1, _seg_frames(sp, dp), endpoint=False):
                pos = sp + t * (dp - sp)
                frames.append({**base, "robot_xy": tuple(pos),
                               "carrying": letter, "board": board_transit})

            for _ in range(ANIM_HOLD):
                frames.append({**base, "robot_xy": tuple(dp),
                               "carrying": None, "board": ba})

        return frames

    @staticmethod
    def _draw_anim_frame(ax, frame: dict, target_word: str, title: str) -> None:
        ax.clear()
        board             = frame["board"]
        position_letter   = board["position_letter"]
        position_occupied = board["position_occupied"]
        wordle_correct    = board["wordle_correct"]
        rx, ry            = frame["robot_xy"]
        carrying          = frame["carrying"]

        for cell_id, (px, py) in enumerate(ALL_POSITIONS):
            color = "#FFB3B3" if cell_id in FORBIDDEN_STAGING_IDS else "#E8E8E8"
            ax.scatter(px, py, s=18, color=color, zorder=1)

        for wi, cid in enumerate(WORDLE_CELL_IDS):
            sx, sy = ALL_POSITIONS[cid]
            ltr    = position_letter[cid]
            color  = "lightgreen" if wordle_correct[wi] else ("salmon" if ltr else "lightyellow")
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
            if cell_id not in WORDLE_CELL_IDS_SET and position_occupied[cell_id]:
                px, py = ALL_POSITIONS[cell_id]
                ltr    = position_letter[cell_id]
                ax.scatter(px, py, s=160, color="steelblue", zorder=3)
                ax.text(px, py, ltr or "?", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white", zorder=4)

        ax.scatter(rx, ry, s=320, color="crimson", marker="D", zorder=6)
        ax.scatter(rx, ry, s=130, color="white",   marker="D", zorder=7)

        if carrying:
            ax.text(rx, ry + 0.32, carrying,
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="crimson",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                              edgecolor="crimson", linewidth=1.5, alpha=0.95),
                    zorder=8)

        step_label = (f"step {frame['step_num']}/{frame['total_steps']}"
                      if "step_num" in frame else "")
        ax.set_xlim(WORKSPACE_X_MIN - 0.5, WORKSPACE_X_MAX + 0.5)
        ax.set_ylim(WORKSPACE_Y_MIN - 0.5, WORKSPACE_Y_MAX + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  [{step_label}]", fontsize=9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(handles=[
            mpatches.Patch(color="lightgreen",  label="Correct slot"),
            mpatches.Patch(color="salmon",      label="Wrong slot"),
            mpatches.Patch(color="steelblue",   label="Staging letter"),
            mpatches.Patch(color="crimson",     label="Robot EE"),
        ], loc="upper right", fontsize=6)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RL task sequencer.")
    parser.add_argument("--no-animate", action="store_true",
                        help="Skip GIF generation (faster runs)")
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    evaluator  = TaskSequencerEvaluator(
        model_path=model_path,
        reward_callback=custom_reward,
    )

    print(f"Model loaded: {model_path}.zip")
    print(f"Holdout test word: {HOLDOUT_WORD}")
    print(f"Animation: {'off' if args.no_animate else 'on'}")

    evaluator.run_evaluation(SCENARIOS, animate=not args.no_animate)
