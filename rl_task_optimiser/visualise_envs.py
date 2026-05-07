"""
Visualise the grid-based Wordle environment layout.

Run:
    python visualise_envs.py

Shows a single annotated figure with:
  • 13×7 grid of 0.75 m cells (board boundary)
  • Wordle zone (y=2.25, x ∈ [-1.5, 1.5]) — 5 yellow cells
  • Forbidden staging zone (C4/C5) — 30 salmon-shaded cells
  • Free staging cells — white
  • Robot home at (0, 0) — blue circle
  • Example letter placements from a C3 reset
  • Dimension annotations and legend
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

from training_env.wordle_env import (
    WordleSequencingEnv,
    ALL_POSITIONS, N_CELLS, N_WORDLE,
    WORDLE_CELL_IDS, WORDLE_CELL_IDS_SET, WORDLE_CELL_ID_TO_IDX,
    FORBIDDEN_STAGING_IDS, OUTER_STAGING_CELLS,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    ROBOT_HOME, GRID_CELL, GRID_COLS, GRID_ROWS, ACTION_DIM, OBS_DIM,
)

TARGET_WORD = "CRANE"
SEED        = 42

_WORDLE_FC    = "#FFFACD"   # light yellow
_WORDLE_EC    = "#333333"
_FORBIDDEN_FC = "#FFB3B3"   # light salmon/red
_FORBIDDEN_EC = "#CC6666"
_FREE_FC      = "#FFFFFF"
_FREE_EC      = "#BBBBBB"
_ROBOT_FC     = "#AED6F1"
_ROBOT_EC     = "#1A5276"
_LETTER_FC    = "#E74C3C"   # tomato red
_BOARD_EC     = "#222222"


def _dummy_reward(**_):
    return 0.0


def _draw_grid(ax):
    for cell_id in range(N_CELLS):
        x, y = ALL_POSITIONS[cell_id]
        hw = GRID_CELL / 2

        if cell_id in WORDLE_CELL_IDS_SET:
            fc, ec, lw, zorder = _WORDLE_FC, _WORDLE_EC, 2.0, 2
        elif cell_id in FORBIDDEN_STAGING_IDS:
            fc, ec, lw, zorder = _FORBIDDEN_FC, _FORBIDDEN_EC, 0.8, 1
        else:
            fc, ec, lw, zorder = _FREE_FC, _FREE_EC, 0.5, 1

        ax.add_patch(patches.Rectangle(
            (x - hw, y - hw), GRID_CELL, GRID_CELL,
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
        ))

        # Light (col, row) label in each cell
        col = cell_id % GRID_COLS
        row = cell_id // GRID_COLS
        ax.text(x, y - hw + 0.08, f"{col},{row}",
                ha="center", va="bottom", fontsize=3.5, color="#AAAAAA", zorder=3)


def _draw_board_boundary(ax):
    hw = GRID_CELL / 2
    ax.add_patch(patches.Rectangle(
        (WORKSPACE_X_MIN - hw, WORKSPACE_Y_MIN - hw),
        WORKSPACE_X_MAX - WORKSPACE_X_MIN + GRID_CELL,
        WORKSPACE_Y_MAX - WORKSPACE_Y_MIN + GRID_CELL,
        linewidth=2.5, edgecolor=_BOARD_EC, facecolor="none", zorder=6,
    ))


def _draw_wordle_labels(ax, target_word):
    for wi, cid in enumerate(WORDLE_CELL_IDS):
        x, y = ALL_POSITIONS[cid]
        ax.text(x, y, target_word[wi],
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="#222222", zorder=5)
        ax.text(x, y + GRID_CELL * 0.42, f"W{wi}",
                ha="center", va="bottom", fontsize=6.5, color="#555555", zorder=5)


def _draw_robot_home(ax):
    ax.add_patch(patches.Circle(
        (ROBOT_HOME[0], ROBOT_HOME[1]), 0.25,
        linewidth=2, edgecolor=_ROBOT_EC, facecolor=_ROBOT_FC, alpha=0.95, zorder=7,
    ))
    ax.text(ROBOT_HOME[0], ROBOT_HOME[1], "⌂",
            ha="center", va="center", fontsize=11, color=_ROBOT_EC, zorder=8)
    ax.text(ROBOT_HOME[0], ROBOT_HOME[1] - 0.42, "HOME\n(0,0)",
            ha="center", va="top", fontsize=6, color=_ROBOT_EC, zorder=8)


def _draw_letters(ax, env):
    for cell_id in range(N_CELLS):
        if env.position_occupied[cell_id] and cell_id not in WORDLE_CELL_IDS_SET:
            x, y = ALL_POSITIONS[cell_id]
            ltr  = env.position_letter[cell_id]
            ax.add_patch(patches.Circle((x, y), 0.27, color=_LETTER_FC, zorder=9))
            ax.text(x, y, ltr,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", zorder=10)


def _draw_annotations(ax):
    hw = GRID_CELL / 2

    # Wordle zone span
    wx0, wy = ALL_POSITIONS[WORDLE_CELL_IDS[0]]
    wx1, _  = ALL_POSITIONS[WORDLE_CELL_IDS[-1]]
    ax.annotate(
        "", xy=(wx1 + hw, wy + 0.78), xytext=(wx0 - hw, wy + 0.78),
        arrowprops=dict(arrowstyle="<->", color=_WORDLE_EC, lw=1.3),
        zorder=11,
    )
    ax.text((wx0 + wx1) / 2, wy + 0.88,
            "Wordle zone  x ∈ [−1.5, 1.5],  y = 2.25",
            ha="center", va="bottom", fontsize=7.5, color=_WORDLE_EC, zorder=11)

    # Forbidden zone brace label
    ax.text(0, 1.5,
            "Forbidden staging zone\n(C4 & C5 — no object placement)",
            ha="center", va="center", fontsize=8, color="#8B0000",
            fontstyle="italic", zorder=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#CC6666"))

    # Board dimension arrow
    ax.annotate(
        "", xy=(WORKSPACE_X_MAX + hw, WORKSPACE_Y_MIN),
        xytext=(WORKSPACE_X_MIN - hw, WORKSPACE_Y_MIN),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0),
        xycoords="data", textcoords="data",
    )
    ax.text(0, WORKSPACE_Y_MIN - 0.55,
            f"Board x: {WORKSPACE_X_MIN - hw:.2f} → {WORKSPACE_X_MAX + hw:.2f} m  "
            f"({GRID_COLS} cols × {GRID_CELL} m)",
            ha="center", va="top", fontsize=7, color="#555555")

    ax.annotate(
        "", xy=(WORKSPACE_X_MAX + hw + 0.15, WORKSPACE_Y_MAX + hw),
        xytext=(WORKSPACE_X_MAX + hw + 0.15, WORKSPACE_Y_MIN - hw),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0),
        xycoords="data", textcoords="data",
    )
    ax.text(WORKSPACE_X_MAX + hw + 0.25, (WORKSPACE_Y_MIN + WORKSPACE_Y_MAX) / 2,
            f"y: {WORKSPACE_Y_MIN - hw:.2f}→{WORKSPACE_Y_MAX + hw:.2f}\n({GRID_ROWS} rows)",
            ha="left", va="center", fontsize=7, color="#555555", rotation=90)


def _add_legend(ax, n_outer, n_forbidden):
    handles = [
        patches.Patch(facecolor=_WORDLE_FC,    edgecolor=_WORDLE_EC,    label=f"Wordle slot ({N_WORDLE})"),
        patches.Patch(facecolor=_FORBIDDEN_FC, edgecolor=_FORBIDDEN_EC, label=f"Forbidden staging zone ({n_forbidden} cells, C4/C5)"),
        patches.Patch(facecolor=_FREE_FC,      edgecolor=_FREE_EC,      label=f"Free grid cell ({n_outer} staging cells)"),
        patches.Patch(facecolor=_ROBOT_FC,     edgecolor=_ROBOT_EC,     label="Robot home (0, 0)"),
        patches.Patch(facecolor=_LETTER_FC,                              label="Letter object (example C3 reset)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.95,
              edgecolor="#AAAAAA")


def _style_axes(ax):
    margin_x = 1.5
    margin_y = 1.0
    ax.set_xlim(WORKSPACE_X_MIN - margin_x, WORKSPACE_X_MAX + margin_x)
    ax.set_ylim(WORKSPACE_Y_MIN - margin_y, WORKSPACE_Y_MAX + margin_y)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_xticks(np.arange(WORKSPACE_X_MIN, WORKSPACE_X_MAX + 0.01, GRID_CELL))
    ax.set_yticks(np.arange(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX + 0.01, GRID_CELL))
    ax.tick_params(labelsize=7)
    ax.grid(False)


def main():
    os.makedirs("logs", exist_ok=True)

    env = WordleSequencingEnv(
        stage=3, reward_callback=_dummy_reward, target_word=TARGET_WORD,
    )
    env.reset(seed=SEED)

    n_outer    = len(OUTER_STAGING_CELLS)
    n_forbidden = len(FORBIDDEN_STAGING_IDS)

    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    fig.suptitle(
        f"Grid-Based Wordle Environment  |  Target: {TARGET_WORD}  |  seed={SEED}\n"
        f"Grid: {GRID_COLS}×{GRID_ROWS} = {N_CELLS} cells, cell size = {GRID_CELL} m  │  "
        f"Action space: {N_CELLS}² = {ACTION_DIM}  │  Obs dim: {OBS_DIM}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    _draw_board_boundary(ax)
    _draw_grid(ax)
    _draw_wordle_labels(ax, TARGET_WORD)
    _draw_robot_home(ax)
    _draw_letters(ax, env)
    _draw_annotations(ax)
    _style_axes(ax)
    _add_legend(ax, n_outer, n_forbidden)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join("logs", "grid_environment_layout.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved  ->  {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
