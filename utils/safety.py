import numpy as np

from termcolor import colored


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
        trajectory_joint_poses: (7, N) array of joint positions, in radians
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


def check_collisions(station, trajectory_joint_poses):
    """
    Check for collisions along a trajectory using the optimization plant.

    Args:
        station: IiwaHardwareStationDiagram instance
        trajectory_joint_poses: (7, N) array of joint positions

    Returns:
        is_valid: bool, True if no collisions detected
        violations: list of waypoint indices where collisions were found
    """
    optimization_plant = station.internal_station.get_optimization_plant()
    optimization_plant_context = (
        station.internal_station.get_optimization_plant_context()
    )
    scene_graph = station.internal_station.get_optimization_diagram_sg()
    scene_graph_context = station.internal_station.get_optimization_diagram_sg_context()

    iiwa_model = optimization_plant.GetModelInstanceByName("iiwa")

    violations = []
    num_waypoints = trajectory_joint_poses.shape[1]

    for j in range(num_waypoints):
        q = trajectory_joint_poses[:, j]
        optimization_plant.SetPositions(optimization_plant_context, iiwa_model, q)

        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

        if query_object.HasCollisions():
            violations.append(j)

    is_valid = len(violations) == 0

    if not is_valid:
        print(
            colored(
                f"⚠ Found collision violations at {len(violations)} waypoints", "yellow"
            )
        )

    return is_valid, violations


def check_safety_constraints(
    station,
    trajectory_joint_poses,
    time_array,
    joint_lower_limits,
    joint_upper_limits,
    max_joint_velocities=np.deg2rad(60 * np.ones(7)),
    checking_joints=True,
    checking_velocities=True,
    checking_collisions=True,
):
    """
    Check all safety constraints for a trajectory.

    Args:
        station: IiwaHardwareStationDiagram instance
        trajectory_joint_poses: (7, N) array of joint positions
        max_joint_velocities: (7,) array of maximum allowed joint velocities (absolute value)

    Returns:
        is_valid: bool, True if all safety constraints are satisfied
        violations: dict containing violation info
    """
    # Check joint limits

    is_valid_limits = True
    is_valid_velocities = True
    is_valid_collisions = True

    if checking_joints:
        is_valid_limits, violations_limits = check_joint_limits(
            trajectory_joint_poses, joint_lower_limits, joint_upper_limits
        )
    else:
        is_valid_limits = True
        violations_limits = []

    # Check joint velocities
    if checking_velocities:
        (
            is_valid_velocities,
            violations_velocities,
            max_recorded_velocity,
        ) = check_joint_velocities(
            trajectory_joint_poses, time_array, max_joint_velocities
        )
    else:
        is_valid_velocities = True
        violations_velocities = []
        max_recorded_velocity = 0.0

    # Check collisions
    if checking_collisions:
        is_valid_collisions, violations_collisions = check_collisions(
            station, trajectory_joint_poses
        )
    else:
        is_valid_collisions = True
        violations_collisions = []

    print(colored("Max recorded velocity: ", "yellow"))
    print(colored(max_recorded_velocity, "yellow"))

    is_valid = is_valid_limits and is_valid_velocities and is_valid_collisions

    violations = {
        "limits": violations_limits,
        "velocities": violations_velocities,
        "collisions": violations_collisions,
    }

    return is_valid, violations
