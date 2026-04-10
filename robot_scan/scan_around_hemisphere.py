# General imports
import argparse
import threading

from enum import Enum, auto
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")  # Use interactive backend
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
    MOVING = auto()
    MOVING_TO_START = auto()
    COMPUTING_IK = auto()
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
    visualize=True,
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


def hemisphere_slerp(A, B, center, radius, speed_factor=0.5):
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


def generate_path_along_hemisphere(
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

    print("t_final for path (slerp): ", t[-1])
    print(f"Number of time points: {len(t)}")
    print(f"Path points shape: {path_points.shape}")
    print(f"Number of rotations: {len(path_rots)}")

    return path_points, path_rots, t


def generate_waypoints_down_optical_axis(
    pose_curr: RigidTransform, num_points: int = 100
):
    """
    Generate waypoints along the optical axis of the current end-effector pose.

    Args:
        pose_curr: RigidTransform of current end-effector pose
        num_points: Number of points to generate along the optical axis

    Returns:
        List of RigidTransform representing the waypoints
    """
    path_points = []
    path_rots = []

    total_distance = 0.05  # Total distance to travel down optical axis

    for i in range(num_points):
        # Move down the optical axis (negative z direction in end-effector frame)
        delta_z = (
            -total_distance * i / num_points
        )  # Move down 10 cm over the course of the path
        delta_transform = RigidTransform(
            np.array([0, 0, delta_z])
        )  # No rotation change, just translation down z-axis
        waypoint = pose_curr @ delta_transform  # Apply the delta to the current pose
        path_points.append(waypoint.translation())
        path_rots.append(waypoint.rotation().matrix())

    path_points = np.array(path_points).T  # Shape (3, num_points)

    return path_points, path_rots, np.linspace(0, 5, num_points)


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


def generate_IK_solutions_for_path(
    path_points, path_rots, kinematics_solver, q_init, elbow_angle
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


def main(use_hardware: bool) -> None:
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

    # ==================================================================
    # Waypoint Generation Setup
    # ==================================================================
    hemisphere_pos = np.array([0.7, 0.0, 0.444444])
    hemisphere_radius = 0.02
    hemisphere_axis = np.array([-1, 0, 0])
    coverage = 0.2  # Fraction of hemisphere to cover

    # Generate waypoints
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=30,
        coverage=coverage,
    )
    scan_idx = 1

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
    microscope_tip_frame = station.get_internal_plant().GetFrameByName(
        "microscope_tip_link"
    )
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=microscope_tip_frame,
        length=0.1,
        radius=0.002,
        name="microscope_link",
    )

    # Build diagram
    diagram = builder.Build()

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Move to Start")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Add custom sliders for latitude and longitude
    station.internal_meshcat.AddSlider("Latitude", -90, 90, 0.1, 0)
    station.internal_meshcat.AddSlider("Longitude", -180, 180, 0.1, 0)

    # region Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station)

    station_context = station.GetMyContextFromRoot(simulator.get_context())

    # Calculate IK for getting to the top of the hemisphere

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    state = State.IDLE

    elbow_angle = np.pi / 2

    # Button management
    num_move_to_top_clicks = 0
    num_execute_traj_clicks = 0

    # IK computation thread state
    ik_thread = None
    ik_result = {"ready": False, "trajectory": None, "trajectory_start_time": None}

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if (
            state
            == State.WAITING_FOR_NEXT_SCAN
            # and station.internal_meshcat.GetButtonClicks("Execute Trajectory") > num_execute_traj_clicks
        ):
            # num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks("Execute Trajectory")

            pose_target = hemisphere_waypoints[scan_idx]

            # Get current end-effector pose from actual robot state
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            internal_plant = station.get_internal_plant()
            internal_plant_context = station.get_internal_plant_context()
            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)

            path_points, path_rots, t = generate_path_along_hemisphere(
                center=hemisphere_pos,
                radius=hemisphere_radius,
                pose_curr=eef_pose,
                pose_target=pose_target,
                hemisphere_axis=hemisphere_axis,
            )

            station.internal_meshcat.SetLine(
                "desired_spiral_path",
                path_points,
                line_width=0.05,
                rgba=Rgba(1, 1, 1, 1),
            )

            # Solve for IK solutions in background thread
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            print(colored("Starting IK computation in background...", "yellow"))

            def compute_ik_async():
                trajectory_joint_poses = generate_IK_solutions_for_path(
                    path_points=path_points,
                    path_rots=path_rots,
                    kinematics_solver=kinematics_solver,
                    q_init=q_curr,
                    elbow_angle=elbow_angle,
                )

                print(f"Trajectory joint poses shape: {trajectory_joint_poses.shape}")
                print(f"Time array length: {len(t)}")
                print(f"Time array: start={t[0]}, end={t[-1]}")
                print(f"Current robot position: {q_curr}")
                print(f"First trajectory position: {trajectory_joint_poses[:, 0]}")
                print(
                    f"Position error at start: {np.linalg.norm(q_curr - trajectory_joint_poses[:, 0])}"
                )

                # Turn into piecewise polynomial trajectory
                traj = PiecewisePolynomial.FirstOrderHold(t, trajectory_joint_poses)
                print(
                    f"Trajectory start_time: {traj.start_time()}, end_time: {traj.end_time()}"
                )

                # Store results
                ik_result["trajectory"] = traj
                ik_result["ready"] = True
                print(colored("✓ IK computation complete!", "green"))

            ik_result["ready"] = False
            ik_thread = threading.Thread(target=compute_ik_async, daemon=True)
            ik_thread.start()
            state = State.COMPUTING_IK

        elif (
            state == State.IDLE
            and station.internal_meshcat.GetButtonClicks("Move to Start") > 0
        ):
            print(colored("Moving to start", "cyan"))

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            target_pose = hemisphere_waypoints[0]
            target_rot = target_pose.rotation().matrix()
            target_pos = target_pose.translation()

            Q = kinematics_solver.IK_for_microscope(
                target_rot, target_pos, psi=elbow_angle
            )

            # Step 2) Find IK closest to current joint values
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )
            q_des = kinematics_solver.find_closest_solution(Q, q_curr)

            print(colored(f"Goal joint configuration for Start: {q_des}", "yellow"))

            start_trajectory = create_traj_from_q1_to_q2(
                station,
                q_current,
                q_des,
            )

            state = State.MOVING_TO_START
            trajectory_start_time = simulator.get_context().get_time()

        elif state == State.MOVING_TO_START:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= start_trajectory.end_time():
                q_desired = start_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.COMPUTING_IK:
            # Wait for IK computation to complete
            if ik_result["ready"]:
                trajectory = ik_result["trajectory"]
                trajectory_start_time = simulator.get_context().get_time()
                print(
                    f"Simulator time when starting trajectory: {trajectory_start_time}"
                )
                print(colored("Starting trajectory execution!", "cyan"))
                scan_idx += 1

                state = State.MOVING

        elif state == State.MOVING:
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

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Start"
        )
        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Move to Start")
    station.internal_meshcat.DeleteButton("Execute Trajectory")
    station.internal_meshcat.DeleteSlider("Latitude")
    station.internal_meshcat.DeleteSlider("Longitude")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
