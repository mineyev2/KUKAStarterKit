"""
Interactive 3D viewer for hemisphere waypoint IK results.
Visualization matches plot_hemisphere_waypoints exactly.

Usage:
    python utils/view_waypoints.py
    python utils/view_waypoints.py --file outputs/waypoints_result.json

Click a point to print its index and position.
"""

import argparse
import json

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data["waypoints"][0], dict):
        positions = np.array([wp["pos"] for wp in data["waypoints"]])
        rotations = np.array([wp["rot"] for wp in data["waypoints"]])
    else:
        # Legacy format: list of positions only, no rotations
        positions = np.array(data["waypoints"])
        rotations = np.tile(np.eye(3), (len(positions), 1, 1))
    failed_indices = set(data["failed_indices"])
    hemisphere_pos = np.array(data["hemisphere_pos"])
    hemisphere_radius = data["hemisphere_radius"]
    hemisphere_axis = np.array(data["hemisphere_axis"])
    return (
        positions,
        rotations,
        failed_indices,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default=str(Path(__file__).parent.parent / "outputs" / "waypoints_result.json"),
    )
    args = parser.parse_args()

    (
        positions,
        rotations,
        failed_indices,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
    ) = load_data(args.file)
    n = len(positions)

    success_idx = [i for i in range(n) if i not in failed_indices]
    fail_idx = sorted(failed_indices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # --- Waypoint scatter (success = black dots, failed = red X) ---
    if success_idx:
        sp = positions[success_idx]
        sc_success = ax.scatter(
            sp[:, 0],
            sp[:, 1],
            sp[:, 2],
            color="black",
            s=5,
            marker=".",
            label="Hemisphere Waypoints",
            picker=5,
        )
    else:
        sc_success = None

    if fail_idx:
        fp = positions[fail_idx]
        sc_fail = ax.scatter(
            fp[:, 0],
            fp[:, 1],
            fp[:, 2],
            color="red",
            s=50,
            marker="x",
            label=f"Failed ({len(fail_idx)})",
            picker=5,
        )
    else:
        sc_fail = None

    # --- Coordinate frames at every waypoint ---
    frame_scale = 0.01
    frame_linewidth = 1.0
    arrow_length_ratio = 0.1
    for i in range(n):
        pos = positions[i]
        R = rotations[i]
        x_axis = R[:, 0] * frame_scale
        y_axis = R[:, 1] * frame_scale
        z_axis = R[:, 2] * frame_scale
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            arrow_length_ratio=arrow_length_ratio,
            linewidth=frame_linewidth,
        )
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="green",
            arrow_length_ratio=arrow_length_ratio,
            linewidth=frame_linewidth,
        )
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            arrow_length_ratio=arrow_length_ratio,
            linewidth=frame_linewidth,
        )

    # --- Sphere surfaces (same as plot_hemisphere_waypoints) ---
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_sphere = hemisphere_pos[0] + hemisphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = hemisphere_pos[1] + hemisphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = hemisphere_pos[2] + hemisphere_radius * np.outer(
        np.ones(np.size(u)), np.cos(v)
    )

    ax.plot_surface(
        x_sphere,
        y_sphere,
        z_sphere,
        alpha=0.1,
        color="lightgray",
        edgecolor="none",
        linewidth=0.3,
    )
    ax.plot_surface(
        x_sphere,
        y_sphere,
        z_sphere,
        alpha=0.2,
        color="cyan",
        edgecolor="black",
        linewidth=0.1,
    )

    # --- Labels / style ---
    ax.set_title("Generated Hemisphere Waypoints")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    # --- Click interaction ---
    annot = ax.text2D(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8),
    )

    def on_pick(event):
        scatter = event.artist
        ind = event.ind[0]
        if scatter is sc_success and success_idx:
            wp_idx = success_idx[ind]
        elif scatter is sc_fail and fail_idx:
            wp_idx = fail_idx[ind]
        else:
            return
        pos = positions[wp_idx]
        status = "FAIL" if wp_idx in failed_indices else "OK"
        msg = f"idx={wp_idx}  [{status}]\n({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        annot.set_text(msg)
        print(f"[click] waypoint {wp_idx} [{status}]  pos={pos.round(4)}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
