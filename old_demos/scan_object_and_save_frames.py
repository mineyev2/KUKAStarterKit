# General imports
import argparse
import queue
import threading

from enum import Enum, auto
from pathlib import Path

import cv2
import matplotlib

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

matplotlib.use("Agg")  # Use interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from mpl_toolkits.mplot3d import Axes3D
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    FrameIndex,
    LogVectorOutput,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
)
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import create_traj_from_q1_to_q2
from utils.kuka_geo_kin import KinematicsSolver


class State(Enum):
    IDLE = auto()
    WAITING_FOR_NEXT_SCAN = auto()
    PLANNING_MOVE_TO_START = auto()
    MOVING_TO_START = auto()
    MOVING_ALONG_HEMISPHERE = auto()
    MOVING_DOWN_OPTICAL_AXIS = auto()
    COMPUTING_IKS = auto()
    PAUSE = auto()
    DONE = auto()


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
    frame_scale = 0.005  # Increased from 0.005 to be more visible
    frame_linewidth = 0.5
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

    # Overlay another hemisphere based on hemisphere_axis
    if np.allclose(hemisphere_axis, [1, 0, 0]):  # x-axis hemisphere
        u = np.linspace(-np.pi / 2, np.pi / 2, 50)
        v = np.linspace(0, np.pi, 50)
    elif np.allclose(hemisphere_axis, [-1, 0, 0]):
        u = np.linspace(np.pi / 2, 3 * np.pi / 2, 50)
        v = np.linspace(0, np.pi, 50)
    elif np.allclose(hemisphere_axis, [0, 1, 0]):  # y-axis hemisphere
        u = np.linspace(0, np.pi, 50)
        v = np.linspace(0, np.pi, 50)  # Only hemisphere where x <= center_x
    elif np.allclose(hemisphere_axis, [0, -1, 0]):
        u = np.linspace(0, np.pi, 50)
        v = np.linspace(-np.pi, 0, 50)  # Only hemisphere where x <= center_x
    elif np.allclose(hemisphere_axis, [0, 0, 1]):  # z-axis hemisphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi / 2, 25)  # Only hemisphere where z <= center_z
    elif np.allclose(hemisphere_axis, [0, 0, -1]):
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(np.pi / 2, np.pi, 25)  # Only hemisphere where z >= center_z

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
            "✓ Hemisphere waypoints generated and saved to outputs/hemisphere_waypoints.png",
            "cyan",
        )
    )


def hemisphere_slerp(A, B, center, radius, speed_factor=1.0):
    """
    Interpolate along the shortest path on a hemisphere between points A and B.

    Parameters:
        A, B: np.array, shape (3,) - start and end points on the hemisphere
        center: np.array, shape (3,) - center of the sphere
        radius: float - radius of the hemisphere
        num_points: int - number of points along the path
        hemisphere_axis: int - axis index defining the hemisphere (default 2 -> z>=0)
        speed_factor: float - factor to scale the speed of traversal along the path (default 0.5 for half speed)

    Returns:
        path: np.array, shape (num_points, 3) - interpolated points on hemisphere
    """
    # Shift to sphere-centered coordinates
    a = A - center
    b = B - center

    # Normalize to unit sphere
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)

    # Compute angle between vectors
    dot = np.clip(np.dot(a_hat, b_hat), -1.0, 1.0)
    theta = np.arccos(dot)

    # Create time array for PiecewisePolynomial
    # Make t depend on the length of the arc that we are traversing for more natural timing (longer paths take more time)
    arc_length = radius * theta
    t_final = (
        arc_length * 125 / speed_factor
    )  # Scale by speed factor to make it faster or slower
    num_points = int(t_final * 500)  # 500Hz update rate seems good

    t = np.linspace(0, t_final, num_points)

    # if theta < 1e-6:  # points are extremely close
    #     path = np.tile(A, (num_points, 1))
    #     return path

    # Slerp interpolation
    t_vals = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 3))
    for i, t_val in enumerate(t_vals):
        path[i] = (
            np.sin((1 - t_val) * theta) * a_hat + np.sin(t_val * theta) * b_hat
        ) / np.sin(theta)

    # Scale and shift back to original sphere
    path = center + radius * path

    # # Enforce hemisphere constraint
    # path[:, hemisphere_axis] = np.maximum(path[:, hemisphere_axis], center[hemisphere_axis])

    return path.T, t


def sphere_frame(p, hemisphere_axis, center):
    """
    Compute a smooth end-effector rotation matrix at point p on a sphere.

    z-axis  -> surface normal
    x-axis  -> projected global reference direction (smooth, no twisting)
    y-axis  -> z cross x

    Parameters
    ----------
    p : array-like (3,)
        Point on the sphere.
    center : array-like (3,)
        Sphere center (default origin).

    Returns
    -------
    R : (3,3) numpy array
        Rotation matrix with columns [x, y, z]
    """

    p = np.asarray(p, dtype=float)
    center = np.asarray(center, dtype=float)

    # Surface normal
    z = p - center
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-9:
        raise ValueError("Point cannot equal sphere center.")
    z = z / z_norm

    # Global reference direction (choose something stable)
    if np.allclose(hemisphere_axis, np.array([1, 0, 0])):
        g = np.array([-1.0, 0.0, 0.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, -1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, 1.0])
    elif np.allclose(hemisphere_axis, np.array([-1, 0, 0])):
        g = np.array([0.0, 0.0, 1.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, -1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, 1.0])
    elif np.allclose(hemisphere_axis, np.array([0, 0, 1])):
        g = np.array([1.0, 0.0, 0.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, -1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, 1.0])
    elif np.allclose(hemisphere_axis, np.array([0, 0, -1])):  # NOTE: Haven't checked
        g = np.array([1.0, 0.0, 0.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, 1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, -1.0])

    # Project g onto tangent plane
    x = g - np.dot(g, z) * z
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-9:
        raise ValueError("Degenerate tangent direction.")
    x = x / x_norm

    # Complete right-handed frame
    y = np.cross(z, x)

    # Ensure orthonormality (numerical cleanup)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)

    R = np.column_stack((x, y, z))

    return R


def generate_hemisphere_waypoints(
    center, radius, hemisphere_axis, coverage=0.2, num_scan_points=30
):
    """
    Generate N approximately uniformly distributed waypoints on a hemisphere.

    Args:
        center: (3,) array of hemisphere center
        radius: float, hemisphere radius
        hemisphere_axis: np.array of shape (3,), axis defining the hemisphere (e.g. [1, 0, 0] for x-axis hemisphere)
        coverage: float between 0 and 1, fraction of hemisphere to cover (default 0.5 for half hemisphere)
        num_scan_points: int, number of waypoints
    """

    waypoints = []
    phi_golden = (1 + np.sqrt(5)) / 2  # golden ratio

    # Normalize hemisphere_axis
    hemisphere_axis = hemisphere_axis / np.linalg.norm(hemisphere_axis)

    for k in range(num_scan_points):
        # Generate points on canonical hemisphere (top at [0, 0, 1])
        z_s = (
            1 - k / (num_scan_points - 1) * coverage
        )  # height from top (1) to equator (0)
        r_xy = np.sqrt(1 - z_s**2)  # radius in xy-plane
        theta = 2 * np.pi * k / phi_golden  # golden angle

        x_s = r_xy * np.cos(theta)
        y_s = r_xy * np.sin(theta)

        # Point on unit hemisphere with top at [0, 0, 1]
        point_canonical = np.array([x_s, y_s, z_s])

        # Compute rotation from [0, 0, 1] to hemisphere_axis
        z_ref = np.array([0.0, 0.0, 1.0])

        # If hemisphere_axis is close to [0, 0, 1], no rotation needed
        if np.allclose(hemisphere_axis, z_ref):
            point_rotated = point_canonical
        # If hemisphere_axis is close to [0, 0, -1], rotate 180 deg around x-axis
        elif np.allclose(hemisphere_axis, -z_ref):
            point_rotated = np.array(
                [point_canonical[0], -point_canonical[1], -point_canonical[2]]
            )
        else:
            # Compute rotation axis (perpendicular to both vectors)
            rotation_axis = np.cross(z_ref, hemisphere_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Compute rotation angle
            cos_angle = np.dot(z_ref, hemisphere_axis)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            # Rodrigues' rotation formula
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            point_rotated = R @ point_canonical

        # Scale by radius and translate by center
        point_world = center + radius * point_rotated

        rotation = RotationMatrix(sphere_frame(point_world, hemisphere_axis, center))
        waypoint = RigidTransform(rotation, point_world)
        waypoints.append(waypoint)

    return waypoints


def generate_poses_along_hemisphere(
    center, radius, pose_curr, pose_target, hemisphere_axis, num_spirals=2
):
    """
    Args:
        - center: (3,) array of hemisphere center
        - radius: radius of hemisphere
        - pose_curr: RigidTransform of current end-effector pose
        - pose_target: RigidTransform of desired end-effector pose on hemisphere
    Returns:
        - path_points: (3, N) array of positions along the path
        - path_rots: List of (3, 3) rotation matrices at each point along the path
    """

    # Step 1: Generate shortest path along hemisphere surface
    A = pose_curr.translation()
    B = pose_target.translation()
    path_points, t = hemisphere_slerp(A, B, center, radius)

    # Generate rotation matrices along the path using the sphere_frame function
    path_rots = []
    num_points = path_points.shape[1]
    for i in range(num_points):
        p = path_points[:, i]
        R = sphere_frame(p, hemisphere_axis, center)
        path_rots.append(R)

    return path_points, path_rots, t


def generate_waypoints_down_optical_axis(
    pose_curr: RigidTransform,
    num_points: int = 100,
    # t_final: float = 2,
    distance: float = 0.025,
):
    """
    Generate waypoints along the optical axis of the current end-effector pose.

    Args:
        pose_curr: RigidTransform of current end-effector pose
        num_points: Number of points to generate along the optical axis
        t_final: Total time to traverse the path (for timing the trajectory)

    Returns:
        List of RigidTransform representing the waypoints
    """
    path_points = []
    path_rots = []

    # Make t_final relative to distance.
    t_final = distance * 2 / 0.025

    for i in range(num_points):
        # Move down the optical axis (negative z direction in end-effector frame)
        delta_z = (
            -distance * i / num_points
        )  # Move down 10 cm over the course of the path
        delta_transform = RigidTransform(
            np.array([0, 0, delta_z])
        )  # No rotation change, just translation down z-axis
        waypoint = pose_curr @ delta_transform  # Apply the delta to the current pose
        path_points.append(waypoint.translation())
        path_rots.append(waypoint.rotation().matrix())

    path_points = np.array(path_points).T  # Shape (3, num_points)

    return path_points, path_rots, np.linspace(0, t_final, num_points)


def find_target_pose_on_hemisphere(center, latitude_deg, longitude_deg, radius):
    """
    Given a hemisphere defined by its center and radius, find the target end-effector pose on the hemisphere surface corresponding to the specified latitude and longitude angles.

    Args:
        center: (x, y, z) coordinates of the hemisphere center
        latitude_deg: Latitude angle in degrees (-90 to 90)
        longitude_deg: Longitude angle in degrees (-180 to 180)
        radius: Radius of the hemisphere
    Returns:
        target_pose: A 4x4 homogeneous transformation matrix representing the desired end-effector pose
    """

    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    x = center[0] - radius * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = center[1] - radius * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = center[2] + radius * np.sin(latitude_rad)

    target_pos = np.array([x, y, z])

    target_rot = sphere_frame(target_pos, center)

    return target_rot, target_pos


def check_joint_limits(trajectory_joint_poses, joint_lower_limits, joint_upper_limits):
    """
    Check if all joint positions in a trajectory are within specified limits.

    Args:
        trajectory_joint_poses: (7, N) array of joint positions
        joint_lower_limits: (7,) array of lower joint limits
        joint_upper_limits: (7,) array of upper joint limits

    Returns:
        is_valid: bool, True if all joints are within limits
        violations: list of dicts containing violation info with keys:
                   'joint_idx', 'waypoint_idx', 'value', 'limit_type', 'limit_value'
    """
    violations = []
    num_joints, num_waypoints = trajectory_joint_poses.shape

    for i in range(num_joints):
        for j in range(num_waypoints):
            joint_value = trajectory_joint_poses[i, j]

            # Check lower limit
            if joint_value < joint_lower_limits[i]:
                violations.append(
                    {
                        "joint_idx": i,
                        "waypoint_idx": j,
                        "value": joint_value,
                        "limit_type": "lower",
                        "limit_value": joint_lower_limits[i],
                        "violation_amount": joint_lower_limits[i] - joint_value,
                    }
                )

            # Check upper limit
            if joint_value > joint_upper_limits[i]:
                violations.append(
                    {
                        "joint_idx": i,
                        "waypoint_idx": j,
                        "value": joint_value,
                        "limit_type": "upper",
                        "limit_value": joint_upper_limits[i],
                        "violation_amount": joint_value - joint_upper_limits[i],
                    }
                )

    is_valid = len(violations) == 0

    if not is_valid:
        print(colored(f"⚠ Found {len(violations)} joint limit violations:", "yellow"))
        for v in violations[:5]:  # Show first 5 violations
            print(
                colored(
                    f"  Joint {v['joint_idx']+1}, waypoint {v['waypoint_idx']}: "
                    f"{np.rad2deg(v['value']):.2f}° violates {v['limit_type']} limit "
                    f"{np.rad2deg(v['limit_value']):.2f}° by {np.rad2deg(v['violation_amount']):.2f}°",
                    "yellow",
                )
            )
        if len(violations) > 5:
            print(colored(f"  ... and {len(violations) - 5} more violations", "yellow"))

    return is_valid, violations


def check_joint_velocities(
    trajectory_joint_poses, t, max_joint_velocities=np.deg2rad(60 * np.ones(7))
):  # Example limits in rad/s
    """
    Check if joint velocities in a trajectory exceed specified limits.

    Args:
        trajectory_joint_poses: (7, N) array of joint positions
        t: (N,) array of time values
        max_joint_velocities: (7,) array of maximum allowed joint velocities (absolute value)

    Returns:
        is_valid: bool, True if all velocities are within limits
        violations: list of dicts containing violation info
        velocities: (7, N-1) array of computed joint velocities
    """
    violations = []
    num_joints, num_waypoints = trajectory_joint_poses.shape

    # Compute velocities using finite differences
    dt = np.diff(t)  # Shape (N-1,)
    dq = np.diff(trajectory_joint_poses, axis=1)  # Shape (7, N-1)
    velocities = dq / dt  # Shape (7, N-1)

    # store max joint velocity across all joints and waypoints for reporting
    max_recorded_velocity = 0.0

    for i in range(num_joints):
        for j in range(velocities.shape[1]):
            vel_abs = np.abs(velocities[i, j])
            max_recorded_velocity = max(max_recorded_velocity, vel_abs)

            if vel_abs > max_joint_velocities[i]:
                violations.append(
                    {
                        "joint_idx": i,
                        "waypoint_idx": j,
                        "velocity": velocities[i, j],
                        "velocity_abs": vel_abs,
                        "limit": max_joint_velocities[i],
                        "violation_amount": vel_abs - max_joint_velocities[i],
                    }
                )

    is_valid = len(violations) == 0

    if not is_valid:
        print(
            colored(f"⚠ Found {len(violations)} joint velocity violations:", "yellow")
        )
        for v in violations[:5]:  # Show first 5 violations
            print(
                colored(
                    f"  Joint {v['joint_idx']+1}, waypoint {v['waypoint_idx']}: "
                    f"velocity {np.rad2deg(v['velocity']):.2f}°/s (abs: {np.rad2deg(v['velocity_abs']):.2f}°/s) "
                    f"exceeds limit {np.rad2deg(v['limit']):.2f}°/s by {np.rad2deg(v['violation_amount']):.2f}°/s",
                    "yellow",
                )
            )
        if len(violations) > 5:
            print(colored(f"  ... and {len(violations) - 5} more violations", "yellow"))

    return is_valid, violations, max_recorded_velocity


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
    Each plot shows 7 subplots (one per joint) showing position vs time.
    Red dotted lines are drawn at joint limits only when the limit falls
    within the plotted data range (so the axis is never artificially expanded).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    joint_names = [f"Joint {i+1}" for i in range(7)]

    # Save hemisphere trajectory
    hemisphere_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    hemisphere_dir.mkdir(parents=True, exist_ok=True)

    fig_h = Figure(figsize=(10, 12))
    FigureCanvasAgg(fig_h)
    axes_h = fig_h.subplots(7, 1, sharex=True)
    fig_h.suptitle(
        f"Hemisphere Trajectory - Scan {scan_idx}", fontsize=14, fontweight="bold"
    )

    for i in range(7):
        data_deg = np.rad2deg(hemisphere_traj[i, :])
        axes_h[i].plot(hemisphere_t, data_deg, linewidth=1.5, color="C0")
        axes_h[i].set_ylabel(f"{joint_names[i]} (deg)", fontsize=10)
        axes_h[i].grid(True, alpha=0.3)
        if i == 6:
            axes_h[i].set_xlabel("Time (s)", fontsize=10)
        # Draw joint limit lines only if the limit is within the data range
        data_min, data_max = data_deg.min(), data_deg.max()
        if joint_lower_limits is not None:
            lo_deg = np.rad2deg(joint_lower_limits[i])
            if data_min <= lo_deg <= data_max:
                axes_h[i].axhline(
                    lo_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )
        if joint_upper_limits is not None:
            hi_deg = np.rad2deg(joint_upper_limits[i])
            if data_min <= hi_deg <= data_max:
                axes_h[i].axhline(
                    hi_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )

    fig_h.tight_layout()
    hemisphere_path = hemisphere_dir / f"scan_{scan_idx:02d}.png"
    fig_h.savefig(hemisphere_path, dpi=150, bbox_inches="tight")
    print(colored(f"✓ Saved hemisphere trajectory to {hemisphere_path}", "cyan"))

    # Save optical axis trajectory
    optical_dir = Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    optical_dir.mkdir(parents=True, exist_ok=True)

    fig_o = Figure(figsize=(10, 12))
    FigureCanvasAgg(fig_o)
    axes_o = fig_o.subplots(7, 1, sharex=True)
    fig_o.suptitle(
        f"Optical Axis Trajectory - Scan {scan_idx}", fontsize=14, fontweight="bold"
    )

    for i in range(7):
        data_deg = np.rad2deg(optical_traj[i, :])
        axes_o[i].plot(optical_t, data_deg, linewidth=1.5, color="C1")
        axes_o[i].set_ylabel(f"{joint_names[i]} (deg)", fontsize=10)
        axes_o[i].grid(True, alpha=0.3)
        if i == 6:
            axes_o[i].set_xlabel("Time (s)", fontsize=10)
        # Draw joint limit lines only if the limit is within the data range
        data_min, data_max = data_deg.min(), data_deg.max()
        if joint_lower_limits is not None:
            lo_deg = np.rad2deg(joint_lower_limits[i])
            if data_min <= lo_deg <= data_max:
                axes_o[i].axhline(
                    lo_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )
        if joint_upper_limits is not None:
            hi_deg = np.rad2deg(joint_upper_limits[i])
            if data_min <= hi_deg <= data_max:
                axes_o[i].axhline(
                    hi_deg, color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )

    fig_o.tight_layout()
    optical_path = optical_dir / f"scan_{scan_idx:02d}.png"
    fig_o.savefig(optical_path, dpi=150, bbox_inches="tight")
    print(colored(f"✓ Saved optical axis trajectory to {optical_path}", "cyan"))


def generate_IK_solutions_for_path(
    path_points,
    path_rots,
    kinematics_solver,
    q_init,
    elbow_angle,
    joint_lower_limits,
    joint_upper_limits,
):
    trajectory_joint_poses = []
    q_prev = (
        q_init  # Try to match first point to current joint configuration for smoothness
    )

    for i in range(len(path_points.T)):
        eef_pos = path_points[:, i]  # Shift spiral to be around the hemisphere center
        eef_rot = path_rots[i]  # Use the rotation matrix from the path
        Q = kinematics_solver.IK_for_microscope(eef_rot, eef_pos, psi=elbow_angle)

        q_curr = kinematics_solver.find_closest_solution(
            Q, q_prev
        )  # Choose closest solution to previous point for smoothness

        trajectory_joint_poses.append(q_curr)
        q_prev = q_curr

    trajectory_joint_poses = np.array(trajectory_joint_poses).T  # Shape (7, num_points)

    return trajectory_joint_poses


def plot_trajectories_side_by_side(
    hemisphere_traj, hemisphere_t, optical_traj, optical_t, scan_idx
):
    """
    Display hemisphere and optical axis trajectories side-by-side in a matplotlib window.
    Left: 7 subplots for hemisphere trajectory (one per joint)
    Right: 7 subplots for optical axis trajectory (one per joint)
    Uses TkAgg backend to ensure window pops up.
    """
    import matplotlib

    matplotlib.use("TkAgg", force=True)
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


def compute_hemisphere_traj_async(
    hemisphere_pos,
    hemisphere_radius,
    hemisphere_axis,
    eef_pose,
    pose_target,
    kinematics_solver,
    q_curr,
    elbow_angle,
    ik_result,
    plot_trajectories=False,
    scan_idx=0,
    joint_lower_limits=None,
    joint_upper_limits=None,
):
    hemisphere_points, hemisphere_rots, hemisphere_t = generate_poses_along_hemisphere(
        center=hemisphere_pos,
        radius=hemisphere_radius,
        pose_curr=eef_pose,
        pose_target=pose_target,
        hemisphere_axis=hemisphere_axis,
    )

    trajectory_joint_poses = generate_IK_solutions_for_path(
        path_points=hemisphere_points,
        path_rots=hemisphere_rots,
        kinematics_solver=kinematics_solver,
        q_init=q_curr,
        elbow_angle=elbow_angle,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
    )

    # Turn into piecewise polynomial trajectory
    traj = PiecewisePolynomial.FirstOrderHold(hemisphere_t, trajectory_joint_poses)
    print(f"Trajectory start_time: {traj.start_time()}, end_time: {traj.end_time()}")

    # Store results (including raw data for plotting)
    ik_result["trajectory"] = traj
    ik_result["trajectory_joint_poses"] = trajectory_joint_poses
    ik_result["time"] = hemisphere_t
    ik_result["scan_idx"] = scan_idx
    ik_result["ready"] = True

    # Generate and save hemisphere trajectory plot (non-blocking, thread-safe)
    if plot_trajectories:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

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

    # Check to make sure velocities are within limits, if not, raise error to trigger replanning with slower speed
    velocities_are_valid, _, max_recorded_velocity = check_joint_velocities(
        trajectory_joint_poses, hemisphere_t
    )
    print(
        colored(
            f"Max recorded joint velocity in hemisphere trajectory: {np.rad2deg(max_recorded_velocity):.2f}°/s",
            "yellow",
        )
    )
    if not velocities_are_valid:
        raise ValueError("joint velocity limit(s) exceeded in hemisphere trajectory")

    joints_are_valid, num_violations = check_joint_limits(
        trajectory_joint_poses, joint_lower_limits, joint_upper_limits
    )

    if not joints_are_valid:
        raise ValueError("joint limit(s) exceeded in hemisphere trajectory")

    print(colored("✓ Hemisphere IK computation complete!", "green"))


def compute_optical_axis_traj_async(
    pose_curr,
    kinematics_solver,
    q_curr,
    elbow_angle,
    ik_result,
    plot_trajectories=False,
    scan_idx=0,
    joint_lower_limits=None,
    joint_upper_limits=None,
    distance: float = 0.025,
):
    path_points, path_rots, t = generate_waypoints_down_optical_axis(
        pose_curr, distance=distance
    )

    trajectory_joint_poses = generate_IK_solutions_for_path(
        path_points=path_points,
        path_rots=path_rots,
        kinematics_solver=kinematics_solver,
        q_init=q_curr,
        elbow_angle=elbow_angle,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
    )

    # # append trajectory to itself but reverese the waypoints to move back up the optical axis
    trajectory_joint_poses = np.hstack(
        (trajectory_joint_poses, trajectory_joint_poses[:, ::-1])
    )
    t = np.hstack((t, t + t[-1] + t[1]))  # Time for returning back up the optical axis

    traj = PiecewisePolynomial.FirstOrderHold(t, trajectory_joint_poses)

    # Store results (including raw data for plotting)
    ik_result["trajectory"] = traj
    ik_result["trajectory_joint_poses"] = trajectory_joint_poses
    ik_result["time"] = t
    ik_result["scan_idx"] = scan_idx
    ik_result["ready"] = True

    # Generate and save optical axis trajectory plot (non-blocking, thread-safe)
    if plot_trajectories:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

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

    # Check to make sure velocities are within limits, if not, raise error to trigger replanning with slower speed
    velocities_are_valid, _, max_recorded_velocity = check_joint_velocities(
        trajectory_joint_poses, t
    )

    print(
        colored(
            f"Max recorded joint velocity in optical axis trajectory: {np.rad2deg(max_recorded_velocity):.2f}°/s",
            "yellow",
        )
    )

    if not velocities_are_valid:
        raise ValueError("joint velocity limit(s) exceeded in optical axis trajectory")

    joints_are_valid, num_violations = check_joint_limits(
        trajectory_joint_poses, joint_lower_limits, joint_upper_limits
    )

    if not joints_are_valid:
        raise ValueError("joint limit(s) exceeded in optical axis trajectory")

    print(colored("✓ Optical axis IK computation complete!", "green"))


def main(use_hardware: bool) -> None:
    # Clean up trajectory output folders
    import shutil

    hemisphere_paths_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    optical_axis_paths_dir = (
        Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    )

    if hemisphere_paths_dir.exists():
        shutil.rmtree(hemisphere_paths_dir)
        print(colored(f"✓ Cleared {hemisphere_paths_dir}", "grey"))

    if optical_axis_paths_dir.exists():
        shutil.rmtree(optical_axis_paths_dir)
        print(colored(f"✓ Cleared {optical_axis_paths_dir}", "grey"))

    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: worldhemisphere_radius
    #     child: sphere_obstacle::sphere_body
    #     X_PC:
    #         translation: [0.5, 0.0, 0.6]
    plant_config:
        # For some reason, this requires a small timestep
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """

    # Clean up trajectory output folders
    import shutil

    hemisphere_paths_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    optical_axis_paths_dir = (
        Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    )

    if hemisphere_paths_dir.exists():
        shutil.rmtree(hemisphere_paths_dir)
        print(colored(f"✓ Cleared {hemisphere_paths_dir}", "grey"))

    if optical_axis_paths_dir.exists():
        shutil.rmtree(optical_axis_paths_dir)
        print(colored(f"✓ Cleared {optical_axis_paths_dir}", "grey"))

    # ==================================================================
    # Waypoint Generation Setup
    # ==================================================================
    hemisphere_pos = np.array([0.5, 0.0, 0.255])
    hemisphere_radius = 0.115
    hemisphere_axis = np.array([0, 0, 1])
    coverage = 0.10  # Fraction of hemisphere to cover
    distance_along_optical_axis = 0.04
    elbow_angle = np.pi / 2 + np.deg2rad(45)

    T_tip_to_camera = np.eye(4)
    T_tip_to_camera[:3, 3] = [0, 0, 0.1]
    T_tip_to_camera[:3, :3] = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ]
    )

    # Generate waypoints
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=30,
        coverage=coverage,
    )
    scan_idx = 1  # Default is 1

    # Plot waypoints for sanity check
    heimsphere_waypoints_output_path = (
        Path(__file__).parent.parent / "outputs" / "hemisphere_waypoints.png"
    )
    plot_hemisphere_waypoints(
        hemisphere_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=heimsphere_waypoints_output_path,
        visualize=True,
    )

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    # Log joint positions using station's exported output port
    from pydrake.systems.primitives import VectorLogSink

    state_logger = builder.AddSystem(VectorLogSink(7))
    state_logger.set_name("state_logger")
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"), state_logger.get_input_port()
    )

    # Create dummy constant position source (using station's default position)
    default_position = station.get_iiwa_controller_plant().GetPositions(
        station.get_iiwa_controller_plant().CreateDefaultContext()
    )
    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Get total number of frames
    camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=camera_frame,
        length=0.1,
        radius=0.002,
        name="camera_link",
    )

    # microscope_tip_frame = station.get_internal_plant().GetFrameByName(
    #     "microscope_tip_link"
    # )
    # AddFrameTriadIllustration(
    #     scene_graph=station.internal_station.get_scene_graph(),
    #     plant=station.get_internal_plant(),
    #     frame=microscope_tip_frame,
    #     length=0.1,
    #     radius=0.002,
    #     name="microscope_tip_link",
    # )

    # Build diagram
    diagram = builder.Build()

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Plan Move to Start")
    station.internal_meshcat.AddButton("Move to Start")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Add joint position sliders (in degrees for readability)
    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    for i in range(7):
        min_deg = np.rad2deg(joint_lower_limits[i])
        max_deg = np.rad2deg(joint_upper_limits[i])
        station.internal_meshcat.AddSlider(
            f"Joint {i+1} (deg)", min_deg, max_deg, 0.1, 0
        )

    # region Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station)

    station_context = station.GetMyContextFromRoot(simulator.get_context())

    # Calculate IK for getting to the top of the hemisphere

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    prev_state = State.IDLE
    state = State.IDLE

    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    # Button management
    num_move_to_top_clicks = 0
    num_compute_iks_clicks = 0
    num_execute_traj_clicks = 0

    # IK computation thread state
    ik_thread = None
    hemisphere_ik_result = {
        "ready": False,
        "trajectory": None,
        "trajectory_start_time": None,
    }
    optical_axis_ik_result = {
        "ready": False,
        "trajectory": None,
        "trajectory_start_time": None,
    }

    # ---------------------------------------------------------------
    # Camera + background thread setup
    #
    # Architecture (zero camera calls on the main loop critical path):
    #
    #   _capture_thread  ←  owns camera.read() loop at camera's native rate
    #           |  enqueues (frame, path) when _recording is set
    #           ↓
    #       _frame_queue
    #           ↓
    #   _writer_thread   ←  cv2.imwrite() calls, fully off critical path
    #
    # The main loop only calls _recording.set() / _recording.clear().
    # ---------------------------------------------------------------
    camera = cv2.VideoCapture(4)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not camera.isOpened():
        print(
            colored(
                "⚠ Could not open camera device 4 – frames will NOT be saved", "yellow"
            )
        )

    # Shared state written by main loop BEFORE _recording.set(); read by capture thread.
    _capture_state = {
        "frame_dir": None,  # Path  – current scan output folder
        "frame_idx": 0,  # int   – frame counter for this scan
    }
    _recording = threading.Event()  # set = capture & enqueue; cleared = discard
    _capture_stop = threading.Event()  # set = capture thread should exit

    _frame_queue: queue.Queue = queue.Queue(maxsize=0)  # unbounded

    def _capture_loop():
        """Runs in background; owns the camera entirely.
        Enqueues (frame, path_str) whenever _recording is set.
        Discards frames when _recording is cleared (keeps camera buffer drained).
        """
        while not _capture_stop.is_set():
            ret, frame = camera.read()  # blocks until next frame at camera rate
            if not ret:
                continue
            if _recording.is_set() and _capture_state["frame_dir"] is not None:
                path = str(
                    _capture_state["frame_dir"]
                    / f"frame_{_capture_state['frame_idx']:05d}.jpg"
                )
                _frame_queue.put((frame, path))
                _capture_state["frame_idx"] += 1

    def _writer_loop():
        """Handles all cv2.imwrite() calls off the critical path."""
        while True:
            item = _frame_queue.get()
            if item is None:  # sentinel – exit
                _frame_queue.task_done()
                break
            frame_data, path_str = item
            cv2.imwrite(path_str, frame_data)
            _frame_queue.task_done()

    _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    _capture_thread.start()
    _writer_thread.start()

    # Per-scan metadata (frame dir / halfway time tracked here; frame_idx lives in _capture_state)
    scan_frame_dir = None  # Path to current scan's frame folder
    optical_halfway_time = None  # traj_time at which inward pass ends

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state != prev_state:
            print(colored(f"State changed: {prev_state} -> {state}", "grey"))
            prev_state = state

        if state == State.IDLE:
            if station.internal_meshcat.GetButtonClicks("Plan Move to Start") > 0:
                print(colored("Planning move to start", "cyan"))

                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )

                target_pose = hemisphere_waypoints[scan_idx - 1]
                target_rot = target_pose.rotation().matrix()
                target_pos = target_pose.translation()

                Q = kinematics_solver.IK_for_microscope(
                    target_rot, target_pos, psi=elbow_angle
                )

                # # print all sols, round to nearest 2nd decimal
                # for i in range(Q.shape[0]):
                #     print(np.rad2deg(Q[i, :]).round(2))

                # Step 2) Find IK closest to current joint values
                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )
                q_des = kinematics_solver.find_closest_solution(Q, q_curr)

                print(
                    colored(
                        f"Goal joint configuration for Start: {np.rad2deg(q_des)}",
                        "yellow",
                    )
                )

                trajectory = create_traj_from_q1_to_q2(
                    station,
                    q_current,
                    q_des,
                )

                state = State.PLANNING_MOVE_TO_START
        elif state == State.PLANNING_MOVE_TO_START:
            if station.internal_meshcat.GetButtonClicks("Move to Start") > 0:
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_START
        elif state == State.WAITING_FOR_NEXT_SCAN:
            # if (
            #     station.internal_meshcat.GetButtonClicks("Compute IKs")
            #     > num_compute_iks_clicks
            # ):
            #     num_compute_iks_clicks = station.internal_meshcat.GetButtonClicks(
            #         "Compute IKs"
            #     )
            # else:
            #     continue

            if scan_idx >= len(hemisphere_waypoints):
                print(colored("✓ All scans complete!", "green"))
                state = State.DONE
                continue

            print(colored(f"Preparing trajectory for scan #{scan_idx}", "grey"))

            pose_curr = hemisphere_waypoints[scan_idx - 1]  # We assume scan_idx >= 1
            pose_target = hemisphere_waypoints[scan_idx]

            # Get current end-effector pose from actual robot state
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            internal_plant = station.get_internal_plant()
            internal_plant_context = station.get_internal_plant_context()
            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            # Setup trajectory IK computations in background threads to avoid blocking the main simulation loop
            print(colored("Starting IK computation in background...", "yellow"))
            hemisphere_ik_result["ready"] = False
            hemisphere_ik_thread = threading.Thread(
                target=compute_hemisphere_traj_async,
                args=(
                    hemisphere_pos,
                    hemisphere_radius,
                    hemisphere_axis,
                    eef_pose,
                    pose_target,
                    kinematics_solver,
                    q_curr,
                    elbow_angle,
                    hemisphere_ik_result,
                    True,  # plot_trajectories
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                ),
                daemon=True,
            )
            hemisphere_ik_thread.start()

            optical_axis_ik_result["ready"] = False
            optical_axis_ik_thread = threading.Thread(
                target=compute_optical_axis_traj_async,
                args=(
                    pose_curr,
                    kinematics_solver,
                    q_curr,
                    elbow_angle,
                    optical_axis_ik_result,
                    True,  # plot_trajectories
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                    distance_along_optical_axis,
                ),
                daemon=True,
            )
            optical_axis_ik_thread.start()

            state = State.COMPUTING_IKS

        elif state == State.COMPUTING_IKS:
            # Wait for IK computation to complete
            if (
                hemisphere_ik_result["ready"]
                and optical_axis_ik_result["ready"]
                and station.internal_meshcat.GetButtonClicks("Execute Trajectory")
                > num_execute_traj_clicks
            ):
                num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
                    "Execute Trajectory"
                )

                hemisphere_trajectory = hemisphere_ik_result["trajectory"]
                optical_axis_trajectory = optical_axis_ik_result["trajectory"]

                # Plotting already happened in the async functions (non-blocking)

                trajectory_start_time = simulator.get_context().get_time()
                print(
                    f"Simulator time when starting trajectory: {trajectory_start_time}"
                )
                print(colored("Starting trajectory execution", "yellow"))

                # ----------------------------------------------------------
                # Compute the halfway point of the optical axis trajectory
                # (the trajectory is mirrored: first half goes inward, second
                # half goes outward).  We only capture frames during the first half.
                # ----------------------------------------------------------
                optical_halfway_time = optical_axis_trajectory.end_time() / 2.0

                # Create per-scan output folder: outputs/scans/scan<NN>/
                scans_base = Path(__file__).parent.parent / "outputs" / "scans"
                scan_frame_dir = scans_base / f"scan{scan_idx:02d}"
                scan_frame_dir.mkdir(parents=True, exist_ok=True)
                midpoint_pose_saved = False  # reset for this scan
                print(colored(f"✓ Frame output dir: {scan_frame_dir}", "cyan"))

                # Wire up capture state BEFORE setting _recording so the
                # capture thread always sees a valid directory and counter.
                _capture_state["frame_dir"] = scan_frame_dir
                _capture_state["frame_idx"] = 0
                _recording.set()  # ← capture thread starts saving frames now

                scan_idx += 1

                state = State.MOVING_DOWN_OPTICAL_AXIS

        elif state == State.MOVING_TO_START:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= trajectory.end_time():
                q_desired = trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                if scan_idx >= len(hemisphere_waypoints):
                    print(colored("✓ All scans complete!", "green"))
                    state = State.DONE
                else:
                    state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_ALONG_HEMISPHERE:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= hemisphere_trajectory.end_time():
                q_desired = hemisphere_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Hemisphere trajectory execution complete!", "green"))
                state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_DOWN_OPTICAL_AXIS:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= optical_axis_trajectory.end_time():
                q_desired = optical_axis_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )

                # -----------------------------------------------------------
                # Stop frame capture once the inward pass is complete.
                # The capture thread does the actual camera.read() + enqueue;
                # clearing the event is all the main loop needs to do.
                # -----------------------------------------------------------
                if traj_time > optical_halfway_time:
                    _recording.clear()

                # -----------------------------------------------------------
                # Save the microscope_tip_link pose at the midpoint of the
                # inward pass (= 1/4 of the total optical axis trajectory).
                # We capture this exactly once per scan (first crossing).
                # -----------------------------------------------------------
                optical_midpoint_time = optical_halfway_time / 2.0  # 1/4 of total traj
                if (
                    not midpoint_pose_saved
                    and scan_frame_dir is not None
                    and traj_time >= optical_midpoint_time
                ):
                    internal_plant = station.get_internal_plant()
                    internal_plant_context = station.get_internal_plant_context()
                    camera_pose = internal_plant.GetFrameByName(
                        "camera_link"
                    ).CalcPoseInWorld(internal_plant_context)
                    pose_matrix = (
                        camera_pose.GetAsMatrix4()
                    )  # 4x4 homogeneous transform

                    pose_path = scan_frame_dir / "pose.npy"
                    np.save(str(pose_path), pose_matrix)
                    midpoint_pose_saved = True
                    print(
                        colored(
                            f"✓ Saved microscope_tip_link pose at t={traj_time:.3f}s → {pose_path}",
                            "cyan",
                        )
                    )

            elif traj_time <= hemisphere_trajectory.end_time() + 1:  # Wait for 1 second
                pass
            else:
                print(colored("✓ Optical axis trajectory execution complete!", "green"))
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_ALONG_HEMISPHERE

        elif state == State.DONE:
            context = simulator.get_context()
            log = state_logger.FindLog(context)
            t = log.sample_times()
            data = log.data()  # shape: (num_states, num_samples)

            # stack time as first column
            out = np.vstack((t, data)).T

            joint_log_csv_path = (
                Path(__file__).parent.parent / "outputs" / "joint_log.csv"
            )
            np.savetxt(
                joint_log_csv_path,
                out,
                delimiter=",",
                header="time," + ",".join([f"x{i}" for i in range(data.shape[0])]),
                comments="",
            )

            # Also save as pickle
            log_data = {"sample_times": log.sample_times(), "data": log.data()}
            import pickle

            joint_log_pkl_path = (
                Path(__file__).parent.parent / "outputs" / "joint_log.pkl"
            )
            with open(joint_log_pkl_path, "wb") as f:
                pickle.dump(log_data, f)
            break
            # pass

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Start"
        )

        # num_compute_iks_clicks = station.internal_meshcat.GetButtonClicks(
        #     "Compute IKs"
        # )

        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        # Update joint sliders to show current joint positions
        station_context = station.GetMyContextFromRoot(simulator.get_context())
        q_current = station.GetOutputPort("iiwa.position_measured").Eval(
            station_context
        )
        for i in range(7):
            station.internal_meshcat.SetSliderValue(
                f"Joint {i+1} (deg)", np.rad2deg(q_current[i])
            )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.005)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Move to Start")
    station.internal_meshcat.DeleteButton("Move to Start")
    station.internal_meshcat.DeleteButton("Execute Trajectory")

    # Delete joint sliders
    for i in range(7):
        station.internal_meshcat.DeleteSlider(f"Joint {i+1} (deg)")

    # Shut down background threads cleanly:
    # 1. Stop the capture thread (it will exit its loop on next camera.read() return)
    _capture_stop.set()
    _recording.clear()  # ensure capture thread drains without enqueuing
    _capture_thread.join(timeout=5)
    camera.release()
    # 2. Drain and stop the writer thread
    _frame_queue.put(None)  # sentinel: tells writer thread to exit
    _frame_queue.join()  # wait for all queued imwrite()s to complete
    _writer_thread.join(timeout=10)
    print(colored("✓ Frame writer flushed and camera released", "grey"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
