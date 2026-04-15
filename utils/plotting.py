"""
utils/plotting.py

Matplotlib helpers for visualizing hemisphere waypoints and trajectories.

Author: Roman Mineyev
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Must be set before importing pyplot
from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Drake imports
from mpl_toolkits.mplot3d import Axes3D
from pydrake.all import RigidTransform, RotationMatrix
from termcolor import colored


def plot_path_with_frames(
    path_points,
    path_rots,
    hemisphere_pos,
    hemisphere_radius,
    output_path,
    frame_scale=0.01,
    num_frames=10,
):
    """
    Plot a 3D path with coordinate frames along it.

    Args:
        path_points: (3, N) array of positions along path
        path_rots: List of (3, 3) rotation matrices at each point
        hemisphere_pos: (3,) array of hemisphere center position
        hemisphere_radius: Radius of hemisphere
        output_path: Path object where to save the figure
        frame_scale: Scale factor for coordinate frame arrows (meters)
        num_frames: Approximate number of frames to display
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw transparent hemisphere sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 25)  # Only upper hemisphere
    x_sphere = hemisphere_pos[0] + hemisphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = hemisphere_pos[1] + hemisphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = hemisphere_pos[2] + hemisphere_radius * np.outer(
        np.ones(np.size(u)), np.cos(v)
    )
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere, alpha=0.2, color="cyan", edgecolor="none"
    )

    # Draw path
    ax.plot(
        path_points[0, :],
        path_points[1, :],
        path_points[2, :],
        label="Hemisphere Path",
        linewidth=2,
    )
    ax.scatter(
        hemisphere_pos[0],
        hemisphere_pos[1],
        hemisphere_pos[2],
        color="red",
        s=100,
        label="Hemisphere Center",
    )

    # Draw coordinate frames along the path (subsample for clarity)
    frame_step = max(1, len(path_rots) // num_frames)
    quiver_length = 0.2
    linewidth = 0.5

    for i in range(0, len(path_rots), frame_step):
        pos = path_points[:, i]
        R = path_rots[i]

        # Extract each axis and scale uniformly
        x_axis = R[:, 0] * frame_scale  # First column
        y_axis = R[:, 1] * frame_scale  # Second column
        z_axis = R[:, 2] * frame_scale  # Third column

        # X axis (red)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            arrow_length_ratio=0.2,
            linewidth=linewidth,
            length=quiver_length,
            normalize=False,
        )
        # Y axis (green)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="green",
            arrow_length_ratio=0.2,
            linewidth=linewidth,
            length=quiver_length,
            normalize=False,
        )
        # Z axis (blue)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            arrow_length_ratio=0.2,
            linewidth=linewidth,
            length=quiver_length,
            normalize=False,
        )

    # Set axis limits based on hemisphere bounds with some padding
    ax.set_xlim(
        hemisphere_pos[0] - hemisphere_radius, hemisphere_pos[0] + hemisphere_radius
    )
    ax.set_ylim(
        hemisphere_pos[1] - hemisphere_radius, hemisphere_pos[1] + hemisphere_radius
    )
    ax.set_zlim(
        hemisphere_pos[2] - hemisphere_radius, hemisphere_pos[2] + hemisphere_radius
    )

    # Set equal aspect ratio for all axes
    # ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Generated Path Along Hemisphere with Coordinate Frames")
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hemisphere_waypoints(
    path_waypoints,
    hemisphere_pos,
    hemisphere_radius,
    hemisphere_axis,
    output_path,
    visualize=False,
):
    # Reconstruct list of points from list of RigidTransforms
    points = np.array([wp.translation() for wp in path_waypoints])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="black",
        s=5,
        marker=".",
        label="Hemisphere Waypoints",
    )

    # Add coordinate frames at all waypoints
    frame_scale = 0.01  # Increased from 0.005 to be more visible
    frame_linewidth = 1.0
    arrow_length_ratio = 0.1
    for wp in path_waypoints:
        pos = wp.translation()
        R = wp.rotation().matrix()

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

    # Add hemisphere surface for visualization, make sure top is at (-R, 0, 0)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)  # Only upper hemisphere
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

    # # Overlay another hemisphere based on hemisphere_axis
    # if np.allclose(hemisphere_axis, [1, 0, 0]):  # x-axis hemisphere
    #     u = np.linspace(-np.pi / 2, np.pi / 2, 50)
    #     v = np.linspace(0, np.pi, 50)
    # elif np.allclose(hemisphere_axis, [-1, 0, 0]):
    #     u = np.linspace(np.pi / 2, 3 * np.pi / 2, 50)
    #     v = np.linspace(0, np.pi, 50)
    # elif np.allclose(hemisphere_axis, [0, 1, 0]):  # y-axis hemisphere
    #     u = np.linspace(0, np.pi, 50)
    #     v = np.linspace(0, np.pi, 50)  # Only hemisphere where x <= center_x
    # elif np.allclose(hemisphere_axis, [0, -1, 0]):
    #     u = np.linspace(0, np.pi, 50)
    #     v = np.linspace(-np.pi, 0, 50)  # Only hemisphere where x <= center_x
    # elif np.allclose(hemisphere_axis, [0, 0, 1]):  # z-axis hemisphere
    #     u = np.linspace(0, 2 * np.pi, 50)
    #     v = np.linspace(0, np.pi / 2, 25)  # Only hemisphere where z <= center_z
    # elif np.allclose(hemisphere_axis, [0, 0, -1]):
    #     u = np.linspace(0, 2 * np.pi, 50)
    #     v = np.linspace(np.pi / 2, np.pi, 25)  # Only hemisphere where z >= center_z

    x_sphere = hemisphere_pos[0] + hemisphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = hemisphere_pos[1] + hemisphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = hemisphere_pos[2] + hemisphere_radius * np.outer(
        np.ones(np.size(u)), np.cos(v)
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

    ax.set_title("Generated Hemisphere Waypoints")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])

    if visualize:
        plt.show()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(
        colored(
            f"✓ Hemisphere waypoints generated and saved to {output_path}",
            "cyan",
        )
    )


def plot_hemisphere_trajectory(
    trajectory_joint_poses,
    hemisphere_t,
    scan_idx,
    joint_lower_limits=None,
    joint_upper_limits=None,
):
    """
    Generate and save hemisphere trajectory plot.
    """
    matplotlib.use("Agg", force=True)

    hemisphere_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    hemisphere_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [f"Joint {i+1}" for i in range(7)]

    fig = Figure(figsize=(10, 12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(7, 1, sharex=True)
    fig.suptitle(
        f"Hemisphere Trajectory - Scan {scan_idx}", fontsize=14, fontweight="bold"
    )

    for i in range(7):
        data_deg = np.rad2deg(trajectory_joint_poses[i, :])
        axes[i].plot(hemisphere_t, data_deg, linewidth=1.5, color="C0")
        axes[i].set_ylabel(f"{joint_names[i]} (deg)", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        if i == 6:
            axes[i].set_xlabel("Time (s)", fontsize=10)
        # Draw joint limit lines only if the limit is within the data range
        data_min, data_max = data_deg.min(), data_deg.max()
        if joint_lower_limits is not None:
            lo_deg = np.rad2deg(joint_lower_limits[i])
            if data_min <= lo_deg <= data_max:
                axes[i].axhline(
                    lo_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )
        if joint_upper_limits is not None:
            hi_deg = np.rad2deg(joint_upper_limits[i])
            if data_min <= hi_deg <= data_max:
                axes[i].axhline(
                    hi_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )

    fig.tight_layout()
    hemisphere_path = hemisphere_dir / f"scan_{scan_idx:02d}.png"
    fig.savefig(hemisphere_path, dpi=150, bbox_inches="tight")
    print(colored(f"✓ Saved hemisphere trajectory to {hemisphere_path}", "cyan"))


def plot_optical_axis_trajectory(
    trajectory_joint_poses,
    t,
    scan_idx,
    joint_lower_limits=None,
    joint_upper_limits=None,
):
    """
    Generate and save optical axis trajectory plot.
    """
    matplotlib.use("Agg", force=True)

    optical_dir = Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    optical_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [f"Joint {i+1}" for i in range(7)]

    fig = Figure(figsize=(10, 12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(7, 1, sharex=True)
    fig.suptitle(
        f"Optical Axis Trajectory - Scan {scan_idx}", fontsize=14, fontweight="bold"
    )

    for i in range(7):
        data_deg = np.rad2deg(trajectory_joint_poses[i, :])
        axes[i].plot(t, data_deg, linewidth=1.5, color="C1")
        axes[i].set_ylabel(f"{joint_names[i]} (deg)", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        if i == 6:
            axes[i].set_xlabel("Time (s)", fontsize=10)
        # Draw joint limit lines only if the limit is within the data range
        data_min, data_max = data_deg.min(), data_deg.max()
        if joint_lower_limits is not None:
            lo_deg = np.rad2deg(joint_lower_limits[i])
            if data_min <= lo_deg <= data_max:
                axes[i].axhline(
                    lo_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )
        if joint_upper_limits is not None:
            hi_deg = np.rad2deg(joint_upper_limits[i])
            if data_min <= hi_deg <= data_max:
                axes[i].axhline(
                    hi_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )

    fig.tight_layout()
    optical_path = optical_dir / f"scan_{scan_idx:02d}.png"
    fig.savefig(optical_path, dpi=150, bbox_inches="tight")
    print(colored(f"✓ Saved optical axis trajectory to {optical_path}", "cyan"))


def save_trajectory_plots(
    hemisphere_traj,
    hemisphere_t,
    optical_traj,
    optical_t,
    scan_idx,
    joint_lower_limits=None,
    joint_upper_limits=None,
):
    """
    Save hemisphere and optical axis trajectory plots to separate files.
    """
    plot_hemisphere_trajectory(
        hemisphere_traj,
        hemisphere_t,
        scan_idx,
        joint_lower_limits,
        joint_upper_limits,
    )
    plot_optical_axis_trajectory(
        optical_traj,
        optical_t,
        scan_idx,
        joint_lower_limits,
        joint_upper_limits,
    )


def plot_trajectories_side_by_side(
    hemisphere_traj, hemisphere_t, optical_traj, optical_t, scan_idx
):
    """
    Display hemisphere and optical axis trajectories side-by-side in a matplotlib window.
    Left: 7 subplots for hemisphere trajectory (one per joint)
    Right: 7 subplots for optical axis trajectory (one per joint)
    Uses TkAgg backend to ensure window pops up.
    """
    # matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 2, figsize=(16, 12), sharex="col")
    fig.suptitle(
        f"Scan {scan_idx} - Hemisphere (Left) vs Optical Axis (Right)",
        fontsize=16,
        fontweight="bold",
    )

    joint_names = [f"Joint {i+1}" for i in range(7)]

    # Plot hemisphere trajectory on left column
    for i in range(7):
        axes[i, 0].plot(
            hemisphere_t, np.rad2deg(hemisphere_traj[i, :]), linewidth=1.5, color="C0"
        )
        axes[i, 0].set_ylabel(f"{joint_names[i]} (deg)", fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title(
                "Hemisphere Trajectory", fontsize=12, fontweight="bold"
            )
        if i == 6:
            axes[i, 0].set_xlabel("Time (s)", fontsize=10)

    # Plot optical axis trajectory on right column
    for i in range(7):
        axes[i, 1].plot(
            optical_t, np.rad2deg(optical_traj[i, :]), linewidth=1.5, color="C1"
        )
        axes[i, 1].set_ylabel(f"{joint_names[i]} (deg)", fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title(
                "Optical Axis Trajectory", fontsize=12, fontweight="bold"
            )
        if i == 6:
            axes[i, 1].set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)  # Allow window to render
