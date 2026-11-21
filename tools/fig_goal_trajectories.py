"""Goal-reaching trajectories of wm_goal.py on antmaze_medium goal_1: 2 checkpoints (rows) x 3 horizons (columns).

Each panel: the maze, the start of every episode, the goal (and its 2 m success radius) and the (x, y) trajectory of each episode.
Reads logs/*WMGOAL-*/{maze,trajectories,episodes}.csv from hpc/ (tools/sync_from_hpc.sh pulls CSVs only).

    python tools/fig_goal_trajectories.py [--logs hpc/logs] [--out figures/goal_traj_antmaze_medium_goal1] [--task goal_1]
Run names: WMGOAL-wm-jepawm_goal1_h<H>-antmaze_medium (JEPA WM) and WMGOAL-fbwm-fbwm_goal1_h<H>-antmaze_medium (FBWM world model).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROWS = [("jepawm", "JEPA world model (reward-free)"), ("fbwm", "FBWM world model")]
HORIZONS = [5, 10, 50]
GOAL_XY = {"goal_1": (20.0, 20.0)}     # antmaze_medium goal_1: cell (6, 6) -> (j*4-4, i*4-4); start cell (1, 1) -> (0, 0)
SUCCESS_RADIUS = 2.0                   # fbwm.py's goal reward
WALL, GOAL, START = "#C4CCD3", "#0064E0", "#1C2B33"


def find_run(logs, key, horizon, task):
    pat = f"{logs}/*WMGOAL-*-{key}_{task.replace('_', '')}_h{horizon}-antmaze_medium"   # goal_1 -> goal1, goal_k25 -> goalk25
    runs = sorted(glob.glob(pat))
    return runs[-1] if runs else None


def draw(ax, run, task, goal_xy):
    maze = pd.read_csv(f"{run}/maze.csv")
    for _, c in maze[maze.wall == 1].iterrows():
        ax.add_patch(Rectangle((c.x - c.unit / 2, c.y - c.unit / 2), c.unit, c.unit, fc=WALL, ec="none"))
    traj = pd.read_csv(f"{run}/trajectories.csv")
    traj = traj[traj.task == task]
    eps = pd.read_csv(f"{run}/episodes.csv")
    eps = eps[eps.task == task]
    cmap = plt.get_cmap("viridis")
    for e, g in traj.groupby("episode"):
        ax.plot(g.x, g.y, lw=1.0, alpha=0.85, color=cmap(e / max(1, traj.episode.max())), zorder=3)
    ax.scatter(eps.start_x, eps.start_y, s=22, marker="o", color=START, zorder=5, label="start")
    gx, gy = goal_xy
    ax.add_patch(Circle((gx, gy), SUCCESS_RADIUS, fc="none", ec=GOAL, lw=1.4, ls="--", zorder=4))
    ax.scatter([gx], [gy], s=110, marker="*", color=GOAL, zorder=6, label="goal")
    ax.set_xlim(maze.x.min() - 2, maze.x.max() + 2)
    ax.set_ylim(maze.y.min() - 2, maze.y.max() + 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="hpc/logs")
    ap.add_argument("--out", default="figures/goal_traj_antmaze_medium_goal1")
    ap.add_argument("--task", default="goal_1")
    ap.add_argument("--goalk", type=int, default=None, help="goal_k runs (wm_goal.py goal_k=K, H=5): 1 x 2 panels, fixed start and goal from pair.csv")
    a = ap.parse_args()
    if a.goalk:
        a.task = f"goal_k{a.goalk}"
        if a.out == "figures/goal_traj_antmaze_medium_goal1":
            a.out = f"figures/goal_traj_antmaze_medium_goalk{a.goalk}"
        HORIZONS[:] = [5]

    if a.goalk:      # one row, one panel per checkpoint
        fig, axs = plt.subplots(1, len(ROWS), figsize=(4.4 * len(ROWS), 4.8))
        axs = np.asarray(axs).reshape(len(ROWS), 1).T
    else:
        fig, axs = plt.subplots(len(ROWS), len(HORIZONS), figsize=(4.2 * len(HORIZONS), 4.4 * len(ROWS)))
    for i, (key, name) in enumerate(ROWS):
        for j, h in enumerate(HORIZONS):
            ax = axs[j, i] if a.goalk else axs[i, j]
            run = find_run(a.logs, key, h, a.task)
            if run is None or not os.path.exists(f"{run}/trajectories.csv"):
                ax.text(0.5, 0.5, "not run yet", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            if a.goalk:
                pair = pd.read_csv(f"{run}/pair.csv").iloc[0]
                goal_xy = (pair.goal_x, pair.goal_y)
            else:
                goal_xy = GOAL_XY[a.task]
            eps = draw(ax, run, a.task, goal_xy)
            ax.set_title(f"{name}, H = {h}\n(x, y) dist. to goal: end {eps.xy_dist_end.mean():.1f}, min {eps.xy_dist_min.mean():.1f}"
                         f"   success {int(eps.score.sum())}/{len(eps)}", fontsize=9)
    axs.flat[0].legend(loc="lower right", fontsize=8, frameon=False)
    what = f"fixed start, goal {a.goalk} steps ahead" if a.goalk else a.task
    fig.suptitle(f"antmaze_medium, {what}: MPPI on the latent distance of the last imagined state to the goal state (10 episodes each)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=150)
    fig.savefig(a.out + ".pdf")
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
