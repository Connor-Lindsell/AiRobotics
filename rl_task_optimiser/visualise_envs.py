"""
Visual inspection of both environment layouts.

Run:
    python visualise_envs.py

Shows both environments side-by-side with:
  • Board boundary
  • Robot home circle (radius 1, centred at origin)
  • Wordle slots (labelled with target-word letters)
  • Letter objects at their sampled positions
  • Complex env: U-shaped staging area with all 11 possible positions shown
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from training_env.simple_env import (
    SimpleWordleEnv,
    SLOT_POSITIONS,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    ROBOT_HOME, ROBOT_RADIUS,
)
from training_env.complex_env import ComplexWordleEnv, U_SHAPE_POSITIONS

TARGET_WORD = "CRANE"
SEED        = 42

# ============================================================
# Dummy callbacks — not needed for visual inspection
# ============================================================

def _dummy_reward(**_):
    return 0.0

def _dummy_obs(object_poses, object_letters, slot_occupied, placed_letters, target_word):
    return np.zeros(25, dtype=np.float32)

# ============================================================
# Drawing helpers
# ============================================================

_BOARD_COLOR    = "#D0D3CF"
_SLOT_COLOR     = "lightyellow"
_ROBOT_COLOR    = "lightblue"
_STAGING_COLOR  = "peachpuff"
_OBJECT_COLOR   = "tomato"


def _draw_board(ax):
    ax.add_patch(patches.Rectangle(
        (WORKSPACE_X_MIN, WORKSPACE_Y_MIN),
        WORKSPACE_X_MAX - WORKSPACE_X_MIN,
        WORKSPACE_Y_MAX - WORKSPACE_Y_MIN,
        linewidth=2, edgecolor="black", facecolor=_BOARD_COLOR, zorder=0,
    ))


def _draw_robot_home(ax):
    ax.add_patch(patches.Circle(
        ROBOT_HOME, ROBOT_RADIUS,
        linewidth=2, edgecolor="dodgerblue", facecolor=_ROBOT_COLOR, alpha=0.8, zorder=2,
    ))
    ax.text(
        ROBOT_HOME[0], ROBOT_HOME[1], "HOME\n(0,0)",
        ha="center", va="center", fontsize=7, zorder=3,
    )


def _draw_slots(ax, target_word):
    for i, (sx, sy) in enumerate(SLOT_POSITIONS):
        ax.add_patch(patches.FancyBboxPatch(
            (sx - 0.5, sy - 0.5), 1.0, 1.0,
            boxstyle="round,pad=0.04",
            linewidth=2, edgecolor="black", facecolor=_SLOT_COLOR, zorder=2,
        ))
        ax.text(sx, sy, target_word[i],
                ha="center", va="center", fontsize=13, fontweight="bold", zorder=3)
        ax.text(sx, sy + 0.62, f"S{i}",
                ha="center", va="bottom", fontsize=7, color="grey", zorder=3)

    # Brace label above the slot cluster
    mid_x = (SLOT_POSITIONS[0][0] + SLOT_POSITIONS[-1][0]) / 2
    top_y = SLOT_POSITIONS[0][1] + 1.0
    ax.text(mid_x, top_y, "Wordle Slots",
            ha="center", va="bottom", fontsize=8, color="dimgrey",
            style="italic", zorder=3)


def _draw_staging_area(ax):
    """Draw all 11 U-shape positions as hollow orange squares."""
    for px, py in U_SHAPE_POSITIONS:
        ax.add_patch(patches.Rectangle(
            (px - 0.45, py - 0.45), 0.9, 0.9,
            linewidth=1.5, edgecolor="darkorange", facecolor=_STAGING_COLOR,
            alpha=0.85, zorder=1,
        ))


def _draw_objects(ax, env):
    if env.object_poses is None:
        return
    for letter, pos in zip(env.object_letters, env.object_poses):
        ax.add_patch(patches.Circle(pos, 0.4, color=_OBJECT_COLOR, zorder=4))
        ax.text(pos[0], pos[1], letter,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", zorder=5)


def _style_axes(ax, title):
    ax.set_xlim(WORKSPACE_X_MIN - 1.0, WORKSPACE_X_MAX + 1.0)
    ax.set_ylim(WORKSPACE_Y_MIN - 1.0, WORKSPACE_Y_MAX + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(range(int(WORKSPACE_X_MIN), int(WORKSPACE_X_MAX) + 1, 2))
    ax.set_yticks(range(int(WORKSPACE_Y_MIN), int(WORKSPACE_Y_MAX) + 1, 2))
    ax.grid(True, alpha=0.25, linestyle="--")


def _add_legend(ax, include_staging=False):
    handles = [
        patches.Patch(facecolor=_ROBOT_COLOR,  edgecolor="dodgerblue", label="Robot home (r=1)"),
        patches.Patch(facecolor=_SLOT_COLOR,   edgecolor="black",      label="Wordle slot (1×1 m)"),
        patches.Patch(facecolor=_OBJECT_COLOR,                          label="Letter object"),
    ]
    if include_staging:
        handles.append(
            patches.Patch(facecolor=_STAGING_COLOR, edgecolor="darkorange", label="Staging position")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

# ============================================================
# Main
# ============================================================

def main():
    os.makedirs("logs", exist_ok=True)

    # Instantiate environments
    simple_env  = SimpleWordleEnv(
        stage=3, reward_callback=_dummy_reward,
        observation_callback=_dummy_obs, target_word=TARGET_WORD,
    )
    complex_env = ComplexWordleEnv(
        stage=3, reward_callback=_dummy_reward,
        observation_callback=_dummy_obs, target_word=TARGET_WORD,
    )

    simple_env.reset(seed=SEED)
    complex_env.reset(seed=SEED)

    # ── Figure ──
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Environment Layouts  |  Target word: {TARGET_WORD}  |  seed={SEED}",
        fontsize=13, fontweight="bold",
    )

    # ── Simple env ──
    _draw_board(ax_s)
    _draw_robot_home(ax_s)
    _draw_slots(ax_s, TARGET_WORD)
    _draw_objects(ax_s, simple_env)
    _style_axes(ax_s, "Simple Environment\n(objects scattered in free workspace)")
    _add_legend(ax_s, include_staging=False)

    # Annotate dimensions
    ax_s.annotate(
        "Board: (−10,0)→(10,10)", xy=(0, -0.6), xycoords="data",
        ha="center", fontsize=7, color="dimgrey",
    )

    # ── Complex env ──
    _draw_board(ax_c)
    _draw_staging_area(ax_c)    # draw U-shape before objects so objects sit on top
    _draw_robot_home(ax_c)
    _draw_slots(ax_c, TARGET_WORD)
    _draw_objects(ax_c, complex_env)
    _style_axes(ax_c, "Complex Environment\n(objects placed in U-shape staging area, ~3 unit gap)")
    _add_legend(ax_c, include_staging=True)

    # Annotate gap arrows on complex env
    ax_c.annotate(
        "", xy=(-2.0, 3.0), xytext=(-5.0, 3.0),
        arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.2),
    )
    ax_c.text(-3.5, 3.15, "gap=3", ha="center", fontsize=7, color="darkorange")

    ax_c.annotate(
        "", xy=(0.0, 3.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.2),
    )
    ax_c.text(0.4, 1.5, "gap=3", ha="left", fontsize=7, color="darkorange")

    plt.tight_layout()

    save_path = os.path.join("logs", "environment_layouts.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved  ->  {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
