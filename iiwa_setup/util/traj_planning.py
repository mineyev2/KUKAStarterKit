import os

from pathlib import Path

import numpy as np

from pydrake.all import (
    BsplineTrajectory,
    GcsTrajectoryOptimization,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Rgba,
    Solve,
)
from termcolor import colored

from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


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


def setup_trajectory_optimization_from_q1_to_q2_without_collision_constraints(
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

    # # --- NEW: Joint Unwrapping ---
    # # Shifting q2 by multiples of 2*pi to find the configuration closest to q1
    # # This prevents the robot from trying to take "the long way around" (e.g. 280 deg jump instead of 80 deg)
    # q2_original = q2.copy()
    # q2 = q1 + (q2 - q1 + np.pi) % (2 * np.pi) - np.pi

    # # Check if we actually shifted anything and print it
    # if not np.allclose(q2, q2_original):
    #     print(colored(f"Adjusted target joints to find closer representation (Unwrapped jump from {np.rad2deg(np.max(np.abs(q1-q2_original))):.1f} to {np.rad2deg(np.max(np.abs(q1-q2))):.1f} deg)", "cyan"))
    # # ----------------------------

    # # Diagnostic: Print joint distance
    # q_diff = np.abs(q1 - q2)
    # max_diff = np.max(q_diff)
    # print(f"DEBUG: Max joint difference between q1 and q2: {np.rad2deg(max_diff):.2f} degrees")
    # if max_diff > 1.5:  # ~90 degrees
    #      print(colored("WARNING: Large joint movement requested! Optimization may be difficult.", "yellow"))

    print("lower vel limits: ", optimization_plant.GetVelocityLowerLimits().flatten())
    print("upper vel limits: ", optimization_plant.GetVelocityUpperLimits().flatten())

    # ============= Costs =============
    trajopt.AddDurationCost(duration_cost)
    trajopt.AddPathLengthCost(path_length_cost)

    # ============= Bounds =============
    trajopt.AddPositionBounds(
        optimization_plant.GetPositionLowerLimits().flatten(),
        optimization_plant.GetPositionUpperLimits().flatten(),
    )
    # trajopt.AddVelocityBounds(
    #     optimization_plant.GetVelocityLowerLimits().flatten(),
    #     optimization_plant.GetVelocityUpperLimits().flatten(),
    # )
    trajopt.AddVelocityBounds(
        -vel_limits.flatten(),
        vel_limits.flatten(),
    )
    # trajopt.AddAccelerationBounds(
    #     -acc_limits.flatten(),
    #     acc_limits.flatten(),
    # )
    # trajopt.AddVelocityBounds(
    #     np.full(num_q, -1.0),
    #     np.full(num_q, 1.0),
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
    # trajopt.AddPathVelocityConstraint(np.zeros(num_q), np.zeros(num_q), 0.0)
    # trajopt.AddPathVelocityConstraint(np.zeros(num_q), np.zeros(num_q), 1.0)

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
    q_safe: np.ndarray = None,
    num_samples: int = 25,
    minimum_distance: float = 0.001,
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

    # # --- NEW: Joint Unwrapping ---
    # # Shifting q2 by multiples of 2*pi to find the configuration closest to q1
    # # This prevents the robot from trying to take "the long way around" (e.g. 280 deg jump instead of 80 deg)
    # q2_original = q2.copy()
    # q2 = q1 + (q2 - q1 + np.pi) % (2 * np.pi) - np.pi

    # # Check if we actually shifted anything and print it
    # if not np.allclose(q2, q2_original):
    #     print(colored(f"Adjusted target joints to find closer representation (Unwrapped jump from {np.rad2deg(np.max(np.abs(q1-q2_original))):.1f} to {np.rad2deg(np.max(np.abs(q1-q2))):.1f} deg)", "cyan"))
    # # ----------------------------

    # ============= Costs =============
    trajopt.AddDurationCost(duration_cost)
    trajopt.AddPathLengthCost(path_length_cost)

    # ============= Bounds =============
    trajopt.AddPositionBounds(
        optimization_plant.GetPositionLowerLimits().flatten(),
        optimization_plant.GetPositionUpperLimits().flatten(),
    )
    # trajopt.AddVelocityBounds(
    #     optimization_plant.GetVelocityLowerLimits().flatten(),
    #     optimization_plant.GetVelocityUpperLimits().flatten(),
    # )
    trajopt.AddVelocityBounds(
        -vel_limits.flatten(),
        vel_limits.flatten(),
    )
    # trajopt.AddAccelerationBounds(
    #     -acc_limits.flatten(),
    #     acc_limits.flatten(),
    # )
    # trajopt.AddVelocityBounds(
    #     np.full(num_q, -1.0),
    #     np.full(num_q, 1.0),
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
    # trajopt.AddPathVelocityConstraint(np.zeros(num_q), np.zeros(num_q), 0.0)
    # trajopt.AddPathVelocityConstraint(np.zeros(num_q), np.zeros(num_q), 1.0)

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

    # ============= INITIAL GUESS ("The Hint") =============
    for i in range(num_control_points):
        s = i / (num_control_points - 1)

        if q_safe is not None:
            # Bend the initial guess through q_safe
            if s <= 0.5:
                phase_s = s / 0.5
                guess_q = q1 + phase_s * (q_safe - q1)
            else:
                phase_s = (s - 0.5) / 0.5
                guess_q = q_safe + phase_s * (q2 - q_safe)
        else:
            # Straight line guess
            guess_q = q1 + s * (q2 - q1)

        prog.SetInitialGuess(trajopt.control_points()[:, i], guess_q)

    # Guess a safe initial duration
    max_vel = optimization_plant.GetVelocityUpperLimits().flatten()
    if q_safe is not None:
        # Distance is q1 -> q_safe -> q2
        dist = np.abs(q_safe - q1) + np.abs(q2 - q_safe)
    else:
        # Distance is just q1 -> q2
        dist = np.abs(q2 - q1)

    guess_duration = np.max(dist / max_vel) * 1.5
    prog.SetInitialGuess(trajopt.duration(), guess_duration)

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

    return trajopt, prog, traj_plot_state
