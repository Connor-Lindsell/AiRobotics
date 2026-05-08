"""
integration_test.py — Pre-ROS2 integration validation for the RL task sequencer.

Simulates the interface the ROS2 node will use:
  - Input : target word (str) + current environment state ({cell_id: letter})
  - Output: ordered pick-and-place task sequence, workspace visualisation,
            and RL-vs-Greedy path length comparison chart

Once validated here, this file migrates into the ROS2 package.
The class mirrors the ROS2 service-server interface exactly.

Cell ID reference (13×7 grid, 91 cells):
  cell_id = row * 13 + col
  x = WORKSPACE_X_MIN + col * 0.75  (cols 0–12, x: -4.5 → +4.5 m)
  y = row * 0.75                     (rows 0–6,  y:  0.0 → +4.5 m)
  Wordle slots: cells 43–47 (row 3, cols 4–8)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from test import (
    TaskSequencerEvaluator,
    WORDLE_CELL_IDS_SET,
    LOGS_DIR,
)
from train import custom_reward, MODEL_DIR, MODEL_NAME
from training_env.wordle_env import (
    ALL_POSITIONS,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
)


# ============================================================
# ROS2 integration interface
# ============================================================

class WorkspaceIntegrationTest:
    """
    Pre-ROS2 integration harness for the RL pick-and-place task sequencer.

    The public interface mirrors what a ROS2 service server will expose:
      - Instantiate once at node startup with the model path.
      - Call run() for each incoming work request (target word + environment state).

    Always uses stage=3: C3 masking handles both C2-like states (empty Wordle
    slots, first task) and C3-like states (some slots occupied by wrong letters,
    subsequent tasks) without needing the caller to specify a stage.
    """

    STAGE = 3

    def __init__(self, model_path: str):
        self.evaluator = TaskSequencerEvaluator(model_path, custom_reward)

    def run(self, requested_work: str, environment_state: dict) -> dict:
        """
        Main entry point — mirrors a ROS2 service call.

        Args:
            requested_work:    Target word to spell, e.g. "CRANE".
            environment_state: {cell_id (int): letter (str)} from the perception
                               system. Includes letters currently in Wordle slots
                               (C3-like) or all letters in staging (C2-like).

        Returns dict with:
            task_sequence   — list of task dicts (ROS2-ready, includes pick/place poses)
            formatted_tasks — list of human-readable task strings
            rl_travel       — total path length by RL agent (metres)
            greedy_travel   — total path length by greedy baseline (metres)
            success         — True if RL agent completed the word
            rl_steps        — number of actions taken by RL
            greedy_steps    — number of actions taken by greedy
            figures         — {"environment": Figure, "comparison": Figure}
        """
        rl_traj, task_sequence = self._build_and_run_rl(requested_work, environment_state)
        greedy_traj             = self._run_greedy(requested_work, environment_state)
        formatted_tasks         = self._format_task_list(task_sequence)
        env_fig                 = self._visualise_environment(rl_traj)
        comp_fig                = self._plot_path_comparison(
            rl_traj["total_travel"], greedy_traj["total_travel"]
        )
        return {
            "task_sequence":   task_sequence,
            "formatted_tasks": formatted_tasks,
            "rl_travel":       rl_traj["total_travel"],
            "greedy_travel":   greedy_traj["total_travel"],
            "success":         rl_traj["success"],
            "rl_steps":        rl_traj["n_steps"],
            "greedy_steps":    greedy_traj["n_steps"],
            "figures":         {"environment": env_fig, "comparison": comp_fig},
        }

    # ----------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------

    def _build_and_run_rl(self, word: str, positions: dict) -> tuple:
        env           = self.evaluator.build_env(stage=self.STAGE, word=word, fixed_positions=positions)
        trajectory    = self.evaluator.run_episode(env)
        task_sequence = self.evaluator.get_task_sequence(trajectory)
        return trajectory, task_sequence

    def _run_greedy(self, word: str, positions: dict) -> dict:
        env = self.evaluator.build_env(stage=self.STAGE, word=word, fixed_positions=positions)
        return self.evaluator.run_episode_greedy(env)

    def _format_task_list(self, task_sequence: list) -> list:
        """
        Format each task as: 'A 01 - [pick: (+x.xx, +y.yy)], [place: (+x.xx, +y.yy)]'

        The letter and step number identify the action at a glance.
        Coordinates use fixed-width fields so columns align in the console output.
        """
        lines = []
        for task in task_sequence:
            letter        = task["letter"]
            step          = task["step"]
            pick_x, pick_y   = task["pick_pose"]
            place_x, place_y = task["place_pose"]
            dest_id       = task["dest_cell_id"]
            dest_label    = f"Wordle slot {task['description'].split('slot ')[-1]}" \
                            if dest_id in WORDLE_CELL_IDS_SET else f"staging cell {dest_id}"
            lines.append(
                f"{letter} {step:02d} - "
                f"[pick: ({pick_x:+.2f}, {pick_y:+.2f})], "
                f"[place: ({place_x:+.2f}, {place_y:+.2f})]"
                f"  ->  {dest_label}"
            )
        return lines

    def _visualise_environment(self, rl_traj: dict) -> plt.Figure:
        """Single-panel workspace figure showing the RL agent's full path."""
        fig, ax  = plt.subplots(figsize=(10, 7))
        word     = rl_traj["target_word"]
        status   = "SUCCESS" if rl_traj["success"] else "FAILED"
        n_steps  = rl_traj["n_steps"]
        travel   = rl_traj["total_travel"]
        title    = f"RL Agent  —  Target: {word}  |  {status}  |  {n_steps} steps  |  {travel:.2f} m"
        self.evaluator.plot_workspace(ax, rl_traj, title)
        fig.tight_layout()
        return fig

    def _plot_path_comparison(self, rl_travel: float, greedy_travel: float) -> plt.Figure:
        """Horizontal bar chart comparing RL vs greedy total path length."""
        fig, ax = plt.subplots(figsize=(7, 3.5))

        labels = ["RL Agent", "Greedy Baseline"]
        values = [rl_travel, greedy_travel]
        colors = ["steelblue", "darkorange"]

        bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)

        max_val = max(values) if max(values) > 0 else 1.0
        for bar, val in zip(bars, values):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} m",
                va="center", fontsize=11, fontweight="bold",
            )

        improvement = (greedy_travel - rl_travel) / greedy_travel * 100 if greedy_travel > 0 else 0.0
        sign        = "+" if improvement >= 0 else ""
        ax.set_xlabel("Total Path Length (m)", fontsize=11)
        ax.set_title(
            f"Path Length: RL vs Greedy\n"
            f"RL is {sign}{improvement:.1f}% {'shorter' if improvement >= 0 else 'longer'} than greedy",
            fontsize=11,
        )
        ax.set_xlim(0, max_val * 1.3)
        ax.axvline(rl_travel,     color="steelblue",  linestyle=":", linewidth=1.2, alpha=0.7)
        ax.axvline(greedy_travel, color="darkorange", linestyle=":", linewidth=1.2, alpha=0.7)
        fig.tight_layout()
        return fig


# ============================================================
# Console helpers
# ============================================================

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_result(label: str, result: dict) -> None:
    print(f"\n  Target word : {label}")
    print(f"  Success     : {'YES' if result['success'] else 'NO'}")
    print(f"  RL steps    : {result['rl_steps']}")
    print(f"  Greedy steps: {result['greedy_steps']}")
    print(f"  RL travel   : {result['rl_travel']:.2f} m")
    print(f"  Greedy trvl : {result['greedy_travel']:.2f} m")
    diff = result['greedy_travel'] - result['rl_travel']
    print(f"  Difference  : {diff:+.2f} m  ({'RL wins' if diff >= 0 else 'Greedy wins'})")
    print(f"\n  Pick-and-place task sequence:")
    for line in result["formatted_tasks"]:
        print(f"    {line}")


# ============================================================
# Demo entry point
# ============================================================

if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_latest")
    runner     = WorkspaceIntegrationTest(model_path)

    TARGET_WORD = "CRANE"

    # ----------------------------------------------------------
    # Scenario A — C2-like
    # All correct letters and one distractor in staging.
    # Wordle slots are empty: represents the FIRST task in a simulation run.
    #
    # Safe outer-row staging cells (not forbidden, not Wordle):
    #   row 0, cols 0–2  → cell IDs 0, 1, 2
    #   row 0, cols 10–12 → cell IDs 10, 11, 12
    # ----------------------------------------------------------
    C2_LIKE_STATE = {
        0:  "C",   # (-4.50, 0.00) — staging
        1:  "R",   # (-3.75, 0.00) — staging
        2:  "A",   # (-3.00, 0.00) — staging
        10: "N",   # ( 3.00, 0.00) — staging
        11: "E",   # ( 3.75, 0.00) — staging
        12: "Z",   # ( 4.50, 0.00) — distractor
    }

    # ----------------------------------------------------------
    # Scenario B — C3-like
    # One Wordle slot has the wrong letter (must be cleared first);
    # another Wordle slot is already correct.
    # Represents a SUBSEQUENT task where some letters are already placed.
    #
    # Wordle slots: cell 43 = slot 0 ('C'), cell 44 = slot 1 ('R'), ...
    # ----------------------------------------------------------
    C3_LIKE_STATE = {
        43: "X",   # Wordle slot 0: wrong letter — must clear before placing 'C'
        44: "R",   # Wordle slot 1: correct  — RL should leave this alone
        0:  "C",   # staging: correct for slot 0
        2:  "A",   # staging: correct for slot 2
        10: "N",   # staging: correct for slot 3
        11: "E",   # staging: correct for slot 4
        12: "Z",   # distractor
    }

    # --- Scenario A ---
    _print_section(f"Scenario A  —  C2-like  (word: {TARGET_WORD}, empty Wordle slots)")
    result_a = runner.run(TARGET_WORD, C2_LIKE_STATE)
    _print_result(TARGET_WORD, result_a)

    result_a["figures"]["environment"].savefig(
        os.path.join(LOGS_DIR, "integration_c2like_environment.png"), dpi=150
    )
    result_a["figures"]["comparison"].savefig(
        os.path.join(LOGS_DIR, "integration_c2like_comparison.png"), dpi=150
    )
    print(f"\n  Figures saved -> logs/integration_c2like_*.png")

    # --- Scenario B ---
    _print_section(f"Scenario B  —  C3-like  (word: {TARGET_WORD}, slot 0 wrong, slot 1 correct)")
    result_b = runner.run(TARGET_WORD, C3_LIKE_STATE)
    _print_result(TARGET_WORD, result_b)

    result_b["figures"]["environment"].savefig(
        os.path.join(LOGS_DIR, "integration_c3like_environment.png"), dpi=150
    )
    result_b["figures"]["comparison"].savefig(
        os.path.join(LOGS_DIR, "integration_c3like_comparison.png"), dpi=150
    )
    print(f"\n  Figures saved -> logs/integration_c3like_*.png")

    print(f"\n{'='*60}")
    print("  Integration test complete. Review figures in logs/.")
    print(f"{'='*60}\n")

    plt.show()
