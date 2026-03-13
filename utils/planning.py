import numpy as np

# Drake
from pydrake.all import (
    KinematicTrajectoryOptimization,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
)
from termcolor import colored

from utils.plotting import plot_hemisphere_trajectory, plot_optical_axis_trajectory
from utils.safety import check_safety_constraints


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
    num_points = int(t_final * 40)

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

    # # Global reference direction (choose something stable)
    # if np.allclose(hemisphere_axis, np.array([1, 0, 0])):
    #     g = np.array([-1.0, 0.0, 0.0])
    #     if np.dot(z, g) > 0.99:
    #         g = np.array([0.0, 0.0, -1.0])
    #     elif np.dot(z, g) < -0.99:
    #         g = np.array([0.0, 0.0, 1.0])
    # elif np.allclose(hemisphere_axis, np.array([-1, 0, 0])):
    #     g = np.array([0.0, 0.0, 1.0])
    #     if np.dot(z, g) > 0.99:
    #         g = np.array([0.0, 0.0, -1.0])
    #     elif np.dot(z, g) < -0.99:
    #         g = np.array([0.0, 0.0, 1.0])
    # elif np.allclose(hemisphere_axis, np.array([0, 0, 1])):
    #     g = np.array([1.0, 0.0, 0.0])
    #     if np.dot(z, g) > 0.99:
    #         g = np.array([0.0, 0.0, -1.0])
    #     elif np.dot(z, g) < -0.99:
    #         g = np.array([0.0, 0.0, 1.0])
    # elif np.allclose(hemisphere_axis, np.array([0, 0, -1])):  # NOTE: Haven't checked
    #     g = np.array([1.0, 0.0, 0.0])
    #     if np.dot(z, g) > 0.99:
    #         g = np.array([0.0, 0.0, 1.0])
    #     elif np.dot(z, g) < -0.99:
    #         g = np.array([0.0, 0.0, -1.0])
    # elif np.allclose(hemisphere_axis, np.array([0, -1, 0])):
    #     g = np.array([0.0, 0.0, 1.0])
    #     if np.dot(z, g) > 0.99:
    #         g = np.array([0.0, 0.0, -1.0])
    #     elif np.dot(z, g) < -0.99:
    #         g = np.array([0.0, 0.0, 1.0])
    # else:
    #     raise ValueError("Invalid hemisphere axis.")

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


def PlotPath(traj_points, station, internal_plant, internal_context):
    """
    Visualize the end-effector path in Meshcat
    """

    cps = traj_points.reshape((7, -1))
    # Reconstruct the spline trajectory
    traj = BsplineTrajectory(trajopt.basis(), cps)
    s_samples = np.linspace(0, 1, 100)
    ee_positions = []
    for s in s_samples:
        q = traj.value(s).flatten()
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
        rgba=traj_plot_state["rgba"],
    )


def setup_trajectory_optimization_from_q1_to_q2(
    station,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,  # Not used currently
    duration_constraints: tuple[float, float],
    num_control_points: int = 10,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    visualize_solving: bool = False,
):
    optimization_plant = station.get_optimization_plant()
    internal_plant = station.get_internal_plant()
    internal_context = station.get_internal_plant_context()
    num_q = optimization_plant.num_positions()

    # dictionary to make it mutable
    traj_plot_state = {"rgba": Rgba(1, 0, 0, 1)}

    print("Planning initial trajectory from q1 to q2")

    trajopt = KinematicTrajectoryOptimization(num_q, num_control_points, spline_order=4)
    prog = trajopt.get_mutable_prog()

    # ============= Costs =============
    trajopt.AddDurationCost(duration_cost)
    trajopt.AddPathLengthCost(path_length_cost)

    # ============= Bounds =============
    trajopt.AddPositionBounds(
        optimization_plant.GetPositionLowerLimits(),
        optimization_plant.GetPositionUpperLimits(),
    )
    trajopt.AddVelocityBounds(
        optimization_plant.GetVelocityLowerLimits(),
        optimization_plant.GetVelocityUpperLimits(),
    )
    trajopt.AddVelocityBounds(
        -vel_limits.reshape((num_q, 1)),
        vel_limits.reshape((num_q, 1)),
    )
    # trajopt.AddAccelerationBounds(
    #     -acc_limits.reshape((num_q, 1)),
    #     acc_limits.reshape((num_q, 1)),
    # )
    # trajopt.AddVelocityBounds(
    #     np.full((num_q, 1), -1.0),
    #     np.full((num_q, 1), 1.0),
    # )

    # ============= Constraints =============
    trajopt.AddDurationConstraint(duration_constraints[0], duration_constraints[1])

    # Position
    trajopt.AddPathPositionConstraint(q1, q1, 0.0)
    trajopt.AddPathPositionConstraint(q2, q2, 1.0)
    # Use quadratic consts to encourage q current and q goal
    prog.AddQuadraticErrorCost(np.eye(num_q), q1, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(np.eye(num_q), q2, trajopt.control_points()[:, -1])

    # Velocity (TOPPRA assumes zero start and end velocities)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

    if visualize_solving:

        def PlotPath(control_points):
            """
            Visualize the end-effector path in Meshcat
            """
            cps = control_points.reshape((num_q, num_control_points))
            # Reconstruct the spline trajectory
            traj = BsplineTrajectory(trajopt.basis(), cps)
            s_samples = np.linspace(0, 1, 100)
            ee_positions = []
            for s in s_samples:
                q = traj.value(s).flatten()
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
                rgba=traj_plot_state["rgba"],
            )

        prog.AddVisualizationCallback(PlotPath, trajopt.control_points().reshape((-1,)))

    return trajopt, prog, traj_plot_state


def add_collision_constraints_to_trajectory(
    station,
    trajopt: KinematicTrajectoryOptimization,
    num_samples: int = 25,
    minimum_distance: float = 0.001,
):
    """
    Add collision avoidance constraints to the trajectory optimization.
    """

    optimization_plant = station.get_optimization_plant()
    optimization_plant_context = (
        station.internal_station.get_optimization_plant_context()
    )

    collision_constraint = MinimumDistanceLowerBoundConstraint(
        optimization_plant,
        minimum_distance,
        optimization_plant_context,
        None,
    )

    evaluate_at_s = np.linspace(0, 1, num_samples)  # TODO: Use a diff value?
    for s in evaluate_at_s:
        trajopt.AddPathPositionConstraint(collision_constraint, s)

    return trajopt


def resolve_with_toppra(
    station,
    trajopt: KinematicTrajectoryOptimization,
    result,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    # Use controller plant because we don't need to check for collisions here
    controller_plant = station.get_iiwa_controller_plant()

    # Reparameterize with TOPPRA
    geometric_path = trajopt.ReconstructTrajectory(result)

    # Diagnostic: Check trajectory properties before TOPPRA
    print("\n=== TOPPRA Diagnostic Info ===")
    print(f"Trajectory duration: {geometric_path.end_time():.4f}s")
    print(f"Velocity limits: {vel_limits}")
    print(f"Acceleration limits: {acc_limits}")

    trajectory = reparameterize_with_toppra(
        geometric_path,
        controller_plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    return trajectory


def create_traj_from_q1_to_q2(
    station,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray = np.full(7, 1.0),
    acc_limits: np.ndarray = np.full(7, 1.0),
    duration_constraints: tuple[float, float] = (0.5, 5.0),
    num_control_points: int = 10,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    visualize_solving: bool = True,
):
    trajopt, prog, traj_plot_state = setup_trajectory_optimization_from_q1_to_q2(
        station,
        q1,
        q2,
        vel_limits,
        acc_limits,
        duration_constraints,
        num_control_points,
        duration_cost,
        path_length_cost,
        visualize_solving,
    )

    # trajopt_with_collisions = add_collision_constraints_to_trajectory(station, trajopt)

    print("Solving trajectory optimization...")
    result = Solve(prog)

    if not result.is_success():
        error_msg = f"Trajectory optimization failed! Solver status: {result.get_solver_id().name()}"
        if result.get_solution_result():
            error_msg += f" - {result.get_solution_result()}"
        raise RuntimeError(error_msg)

    print("Trajectory optimization succeeded!")

    trajectory = resolve_with_toppra(station, trajopt, result, vel_limits, acc_limits)

    print(f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s")

    return trajectory
