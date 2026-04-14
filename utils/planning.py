import os
import time

import numpy as np

from requests import options

os.environ["MOSEKLM_LICENSE_FILE"] = "/home/rmineyev3/mosek/mosek.lic"

# Drake
from pydrake.all import (
    BsplineTrajectory,
    GcsTrajectoryOptimization,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    KinematicTrajectoryOptimization,
    LoadIrisRegionsYamlFile,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Point,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Solve,
    Sphere,
)
from pydrake.solvers import MosekSolver, SnoptSolver
from termcolor import colored

from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from utils.iris import compute_iris_regions
from utils.plotting import plot_hemisphere_trajectory, plot_optical_axis_trajectory
from utils.safety import check_safety_constraints
from utils.states import State


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

    # Handle degenerate case: A and B are the same point (or nearly so)
    if theta < 1e-6:
        t_final = 0.5 / speed_factor
        t = np.linspace(0, t_final, 3)
        path = np.tile(A, (3, 1))
        return path.T, t

    # Create time array for PiecewisePolynomial
    # Make t depend on the length of the arc for more natural timing
    arc_length = radius * theta
    t_final = arc_length * 125 / speed_factor
    num_points = max(3, int(t_final * 40))

    t = np.linspace(0, t_final, num_points)

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

    g = np.array([0.0, 0.0, 1.0])
    if np.dot(z, g) > 0.99:
        g = np.array([0.0, 0.0, -1.0])
    elif np.dot(z, g) < -0.99:
        g = np.array([0.0, 0.0, 1.0])

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
    center,
    radius,
    hemisphere_axis,
    coverage=0.2,
    num_scan_points=30,
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

        # add custom rotation
        additional_rotation = RotationMatrix(
            np.array(
                [
                    [np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0],
                    [np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0],
                    [0, 0, 1],
                ]
            )
        )  # -90 deg rotation around z-axis

        rotation = RotationMatrix(sphere_frame(point_world, hemisphere_axis, center))
        # rotation = additional_rotation @ rotation
        waypoint = RigidTransform(rotation, point_world)
        waypoints.append(waypoint)

    return waypoints


def generate_poses_along_hemisphere(
    center, radius, pose_curr, pose_target, hemisphere_axis, speed_factor=1.0
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
    path_points, t = hemisphere_slerp(A, B, center, radius, speed_factor=speed_factor)

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
    speed_factor: float = 1.0,
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
    t_final = (distance * 2 / 0.025) / speed_factor

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


def compute_hemisphere_traj_async(
    station,
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
    speed_factor=1.0,
    max_joint_velocities=None,
    save_path=None,
):
    hemisphere_points, hemisphere_rots, hemisphere_t = generate_poses_along_hemisphere(
        center=hemisphere_pos,
        radius=hemisphere_radius,
        pose_curr=eef_pose,
        pose_target=pose_target,
        hemisphere_axis=hemisphere_axis,
        speed_factor=speed_factor,
    )

    # print("Number of points in path: ", len(hemisphere_points.T))

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
    traj = PiecewisePolynomial.CubicShapePreserving(
        hemisphere_t, trajectory_joint_poses
    )
    print(f"Trajectory start_time: {traj.start_time()}, end_time: {traj.end_time()}")

    # Store results (including raw data for plotting)
    ik_result["trajectory"] = traj
    ik_result["trajectory_joint_poses"] = trajectory_joint_poses
    ik_result["time"] = hemisphere_t
    ik_result["scan_idx"] = scan_idx

    # Generate and save hemisphere trajectory plot (non-blocking, thread-safe)
    if plot_trajectories:
        plot_hemisphere_trajectory(
            trajectory_joint_poses,
            hemisphere_t,
            scan_idx,
            joint_lower_limits,
            joint_upper_limits,
        )

    # Check safety constraints
    is_safe, violations = check_safety_constraints(
        station,
        trajectory_joint_poses,
        hemisphere_t,
        joint_lower_limits,
        joint_upper_limits,
        max_joint_velocities,
        save_path=save_path,
    )

    ik_result["valid_joints"] = len(violations["limits"]) == 0
    ik_result["valid_velocities"] = len(violations["velocities"]) == 0
    ik_result["valid_collisions"] = len(violations["collisions"]) == 0
    ik_result["ready"] = True

    # if not is_safe:
    #     return

    print(colored("✓ Hemisphere IK computation complete!", "green"))


def compute_optical_axis_traj_async(
    station,
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
    speed_factor: float = 1.0,
    max_joint_velocities=None,
):
    path_points, path_rots, t = generate_waypoints_down_optical_axis(
        pose_curr, distance=distance, speed_factor=speed_factor
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

    traj = PiecewisePolynomial.CubicShapePreserving(t, trajectory_joint_poses)

    # Store results (including raw data for plotting)
    ik_result["trajectory"] = traj
    ik_result["trajectory_joint_poses"] = trajectory_joint_poses
    ik_result["time"] = t
    ik_result["scan_idx"] = scan_idx

    # Generate and save optical axis trajectory plot (non-blocking, thread-safe)
    if plot_trajectories:
        plot_optical_axis_trajectory(
            trajectory_joint_poses,
            t,
            scan_idx,
            joint_lower_limits,
            joint_upper_limits,
        )

    # Check safety constraints
    is_safe, violations = check_safety_constraints(
        station,
        trajectory_joint_poses,
        t,
        joint_lower_limits,
        joint_upper_limits,
        max_joint_velocities,
        checking_collisions=False,
    )

    ik_result["valid_joints"] = len(violations["limits"]) == 0
    ik_result["valid_velocities"] = len(violations["velocities"]) == 0
    ik_result["valid_collisions"] = len(violations["collisions"]) == 0
    ik_result["ready"] = True

    # if not is_safe:
    #     return

    print(colored("✓ Optical axis IK computation complete!", "green"))


def compute_simple_traj_from_q1_to_q2(
    plant,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    print("Generating simple trajectory from q1 to q2")
    path = PiecewisePolynomial.FirstOrderHold([0, 1], np.column_stack((q1, q2)))

    print("Updating with TOPPRA to enforce velocity and acceleration limits")
    traj = reparameterize_with_toppra(
        path,
        plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    print("Trajectory generation complete!")
    return traj


def _draw_positions_as_spheres(meshcat, positions, name, rgba, radius=0.004):
    """
    Draw a list of 3D positions as spheres in meshcat.
    Clears the parent path first so stale dots from a previous call are removed.
    """
    meshcat.Delete(name)
    sphere = Sphere(radius)
    for i, pos in enumerate(positions):
        meshcat.SetObject(f"{name}/{i}", sphere, rgba)
        meshcat.SetTransform(f"{name}/{i}", RigidTransform(pos))


def plot_trajectory_in_meshcat(
    station,
    trajectory,
    rgba=Rgba(0, 1, 0, 1),
    name="gcs_path",
    num_samples=100,
    radius=0.004,
):
    """
    Visualize a Drake Trajectory in Meshcat by sampling FK and drawing spheres.
    """
    internal_plant = station.get_internal_plant()
    internal_context = station.get_internal_plant_context()

    if trajectory is None:
        print(f"Warning: Trajectory '{name}' is None, skipping visualization.")
        return

    try:
        t_start = trajectory.start_time()
        t_end = trajectory.end_time()
    except Exception as e:
        print(f"Warning: Could not get start/end time for trajectory '{name}': {e}")
        return

    if t_start >= t_end:
        print(f"Warning: Trajectory '{name}' has zero or negative duration, skipping.")
        return

    positions = []
    for t in np.linspace(t_start, t_end, num_samples):
        q = trajectory.value(t).flatten()
        internal_plant.SetPositions(internal_context, q)
        X_WB = internal_plant.EvalBodyPoseInWorld(
            internal_context, internal_plant.GetBodyByName("microscope_tip_link")
        )
        positions.append(X_WB.translation())

    _draw_positions_as_spheres(station.internal_meshcat, positions, name, rgba, radius)


def plot_configs_in_meshcat(
    station,
    configs,
    rgba=Rgba(1, 0.5, 0.0, 1),
    name="configs_path",
    radius=0.004,
):
    """
    Visualize a list of joint configurations as spheres in meshcat.
    Useful for displaying an initial guess or any discrete set of configs.
    """

    internal_plant = station.get_internal_plant()
    internal_context = station.get_internal_plant_context()

    positions = []
    for q in configs:
        internal_plant.SetPositions(internal_context, q)
        X_WB = internal_plant.EvalBodyPoseInWorld(
            internal_context, internal_plant.GetBodyByName("microscope_tip_link")
        )
        positions.append(X_WB.translation())

    _draw_positions_as_spheres(station.internal_meshcat, positions, name, rgba, radius)


def PlotPath(
    traj_points, station, internal_plant, internal_context, rgba=Rgba(1, 0, 0, 1)
):
    """
    Visualize the end-effector path in Meshcat from a set of control points.
    """
    num_q = internal_plant.num_positions()
    num_control_points = traj_points.size // num_q
    cps = traj_points.reshape((num_q, num_control_points))

    # Reconstruct a simple linear path for visualization of control points if needed,
    # or follow the spline logic.
    # For a general PlotPath, we'll assume we want to sample a line between them
    # or just use the points directly if they represent samples.

    ee_positions = []
    for i in range(num_control_points):
        q = cps[:, i]
        internal_plant.SetPositions(internal_context, q)
        X_WB = internal_plant.EvalBodyPoseInWorld(
            internal_context,
            internal_plant.GetBodyByName("microscope_tip_link"),
        )
        ee_positions.append(X_WB.translation())

    ee_positions = np.array(ee_positions).T  # shape (3, N)
    station.internal_meshcat.SetLine(
        "positions_path",
        ee_positions,
        line_width=0.05,
        rgba=rgba,
    )


def setup_gcs_traj_opt_from_q1_to_q2(
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    regions=None,
):
    gcs = GcsTrajectoryOptimization(len(q1))

    start = gcs.AddRegions([Point(q1)], order=0)
    goal = gcs.AddRegions([Point(q2)], order=0)

    print(colored("Added start and goal to GCS...", "grey"))

    main = gcs.AddRegions(
        regions=regions,
        order=3,
        h_min=0.1,  # NOTE: Default is 1e-6
        h_max=20.0,
        name="main",
    )

    print(colored(f"Added {len(regions)} Iris regions to GCS...", "grey"))

    gcs.AddEdges(start, main)
    gcs.AddEdges(main, goal)

    print(colored("Added edges to GCS...", "grey"))

    main.AddTimeCost(duration_cost)
    main.AddPathLengthCost(path_length_cost)

    print(colored("Added costs to GCS...", "grey"))

    # main.AddVelocityBounds(
    #     optimization_plant.GetVelocityLowerLimits().flatten(),
    #     optimization_plant.GetVelocityUpperLimits().flatten(),
    # )

    # print(colored("Added velocity bounds to GCS...", "grey"))

    main.AddContinuityConstraints(
        1
    )  # C1 continuity for smoothness (position and velocity continuity)
    main.AddContinuityConstraints(2)  # C2 continuity for smooth acceleration profiles

    print(colored("Added continuity constraints to GCS...", "grey"))

    return gcs, start, goal


def resolve_gcs_with_toppra(
    station,
    raw_trajectory,  # Pass the trajectory returned from gcs.SolvePath()
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    # Use controller plant because we don't need to check for collisions here
    controller_plant = station.get_iiwa_controller_plant()

    # 1. Strip the GCS timings to create a purely geometric path r(s)
    # The "time" variable now simply maps to the number of segments (1 unit per segment)
    geometric_path = GcsTrajectoryOptimization.NormalizeSegmentTimes(raw_trajectory)

    # Diagnostic: Check trajectory properties before TOPPRA
    print("\n=== TOPPRA Diagnostic Info ===")
    print(f"Geometric path segment count: {geometric_path.end_time():.4f}")
    print(f"Velocity limits: {vel_limits}")
    print(f"Acceleration limits: {acc_limits}")

    # 2. Reparameterize with TOPPRA to apply physical timing
    trajectory = reparameterize_with_toppra(
        geometric_path,
        controller_plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    return trajectory


def solve_gcs_traj_opt(
    station,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    compute_iris: bool = False,
):
    """

    Returns:
        trajectory: PiecewisePolynomial
        success: bool (True if optimization succeeded)
    """

    if compute_iris:
        regions = list(compute_iris_regions(station).values())
    else:
        regions = list(LoadIrisRegionsYamlFile("iris_regions_85.yaml").values())

    # Debugging
    q1_valid = -1
    q2_valid = -1
    for i, region in enumerate(regions):
        # print(f"Region {i}: contains q1={region.PointInSet(q1)}, contains q2={region.PointInSet(q2)}")
        if region.PointInSet(q1):
            q1_valid = i
        if region.PointInSet(q2):
            q2_valid = i

    if (q1_valid == -1) or (q2_valid == -1):
        print(
            colored("❌ Warning: q1 or q2 is not contained in any IRIS region!", "red")
        )
    else:
        print(
            colored(
                f"✓ q1 is contained in region {q1_valid} and q2 is contained in region {q2_valid}",
                "green",
            )
        )

    # 1) Setup GCS optimization problem
    gcs, start, goal = setup_gcs_traj_opt_from_q1_to_q2(
        q1=q1,
        q2=q2,
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        duration_cost=1.0,
        path_length_cost=1.0,
        regions=regions,
    )

    # 2) Solve optimization problem
    options = GraphOfConvexSetsOptions()
    options.max_rounded_paths = 5  # default is 5
    options.solver = MosekSolver()
    options.restriction_solver = SnoptSolver()

    # print time taken to solve
    start_time = time.time()
    trajectory, result = gcs.SolvePath(start, goal, options)
    end_time = time.time()
    print(f"GCS solve time: {end_time - start_time:.4f} seconds")

    # Quit if invalid
    if not result.is_success():
        print(colored("❌ GCS shortest path failed!", "red"))
        return None, False

    print(colored("✓ GCS trajectory optimization succeeded!", "green"))

    # 3) Reparameterize with TOPPRA
    trajectory = resolve_gcs_with_toppra(station, trajectory, vel_limits, acc_limits)

    print(colored("✓ TOPPRA succeeded!", "green"))
    return trajectory, True


def solve_gcs_traj_opt_async(
    station,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    result_dict: dict,
    compute_iris: bool = False,
):
    """
    Wrapper for solve_gcs_traj_opt to be used in a background thread.
    """
    trajectory, success = solve_gcs_traj_opt(
        station=station,
        q1=q1,
        q2=q2,
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        compute_iris=compute_iris,
    )

    result_dict["trajectory"] = trajectory
    result_dict["success"] = success
    result_dict["ready"] = True


def move_along_trajectory(traj, start_time, simulator, station):
    # current_time = simulator.get_context().get_time()
    # traj_time = current_time - trajectory_start_time

    # if traj_time <= initial_trajectory.end_time():
    #     q_desired = initial_trajectory.value(traj_time)
    #     station_context = station.GetMyMutableContextFromRoot(
    #         simulator.get_mutable_context()
    #     )
    #     station.GetInputPort("iiwa.position").FixValue(
    #         station_context, q_desired
    #     )
    # else:
    #     print(colored("✓ Trajectory execution complete!", "green"))
    #     if scan_idx >= len(hemisphere_waypoints):
    #         print(colored("✓ All scans complete!", "green"))
    #         state = State.DONE
    #     else:
    #         state = State.WAITING_FOR_NEXT_SCAN

    current_time = simulator.get_context().get_time()
    traj_time = current_time - start_time
    traj_complete = traj_time > traj.end_time()

    if not traj_complete:
        q_desired = traj.value(traj_time)
        station_context = station.GetMyMutableContextFromRoot(
            simulator.get_mutable_context()
        )
        station.GetInputPort("iiwa.position").FixValue(station_context, q_desired)
    else:
        print(colored("✓ Trajectory execution complete!", "green"))

    return traj_complete


def wait_for_trajectory_plan(thread_dict, station):
    """Wait for the background thread to finish computing the trajectory, then visualize it and update state."""

    if thread_dict["ready"]:
        if thread_dict["success"]:
            prescan_trajectory = thread_dict["trajectory"]

            plot_configs_in_meshcat(
                station,
                thread_dict["guess_qs"],
                name="guess_traj",
            )

            plot_trajectory_in_meshcat(
                station,
                prescan_trajectory,
                name="final_traj",
            )

            print(
                colored(
                    "✓ GCS planning for start move complete. Moving now...",
                    "green",
                )
            )
        else:
            print(colored("❌ GCS planning failed!", "red"))
            quit()

    return thread_dict["ready"] and thread_dict["success"]
