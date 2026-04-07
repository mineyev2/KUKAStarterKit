import numpy as np

from pydrake.all import (
    BsplineTrajectory,
    GcsTrajectoryOptimization,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
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
):
    optimization_plant = station.get_optimization_plant()
    num_q = optimization_plant.num_positions()

    print("Planning initial trajectory from q1 to q2")

    trajopt = KinematicTrajectoryOptimization(num_q, num_control_points, spline_order=4)
    prog = trajopt.get_mutable_prog()

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
    trajopt.AddVelocityBounds(
        -vel_limits.flatten(),
        vel_limits.flatten(),
    )

    # ============= Constraints =============
    trajopt.AddDurationConstraint(duration_constraints[0], duration_constraints[1])

    # Position
    trajopt.AddPathPositionConstraint(q1, q1, 0.0)
    trajopt.AddPathPositionConstraint(q2, q2, 1.0)
    # Use quadratic consts to encourage q current and q goal
    prog.AddQuadraticErrorCost(np.eye(num_q), q1, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(np.eye(num_q), q2, trajopt.control_points()[:, -1])

    return trajopt, prog


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

    evaluate_at_s = np.linspace(0, 1, num_samples)
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

    geometric_path = trajopt.ReconstructTrajectory(result)

    print("\n=== TOPPRA Diagnostic Info ===")
    print(f"Trajectory duration: {geometric_path.end_time():.4f}s")
    print(f"Velocity limits: {vel_limits}")
    print(f"Acceleration limits: {acc_limits}")

    trajectory, toppra_success = reparameterize_with_toppra(
        geometric_path,
        controller_plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
        num_grid_points=100,
    )

    return trajectory, toppra_success, geometric_path


def resolve_gcs_with_toppra(
    station,
    raw_trajectory,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
):
    controller_plant = station.get_iiwa_controller_plant()

    geometric_path = GcsTrajectoryOptimization.NormalizeSegmentTimes(raw_trajectory)

    print("\n=== TOPPRA Diagnostic Info ===")
    print(f"Geometric path segment count: {geometric_path.end_time():.4f}")
    print(f"Velocity limits: {vel_limits}")
    print(f"Acceleration limits: {acc_limits}")

    trajectory = reparameterize_with_toppra(
        geometric_path,
        controller_plant,
        velocity_limits=vel_limits,
        acceleration_limits=acc_limits,
    )

    return trajectory


def setup_trajectory_optimization_from_q1_to_q2(
    station,
    q1: np.ndarray,
    q_safe: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,  # Not used currently
    duration_constraints: tuple[float, float],
    num_control_points: int = 10,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    num_samples: int = 25,
    minimum_distance: float = 0.001,
):
    optimization_plant = station.get_optimization_plant()
    num_q = optimization_plant.num_positions()

    print("Planning initial trajectory from q1 to q2")
    print("q safe: ", q_safe)

    trajopt = KinematicTrajectoryOptimization(num_q, num_control_points, spline_order=4)
    prog = trajopt.get_mutable_prog()

    # ============= Costs =============
    # trajopt.AddDurationCost(duration_cost)
    # trajopt.AddPathLengthCost(path_length_cost)

    # ============= Bounds =============
    trajopt.AddPositionBounds(
        optimization_plant.GetPositionLowerLimits().flatten(),
        optimization_plant.GetPositionUpperLimits().flatten(),
    )
    # trajopt.AddVelocityBounds(
    #     -vel_limits.flatten(),
    #     vel_limits.flatten(),
    # )

    # ============= Constraints =============
    trajopt.AddDurationConstraint(duration_constraints[0], duration_constraints[1])

    # Position
    trajopt.AddPathPositionConstraint(q1, q1, 0.0)
    trajopt.AddPathPositionConstraint(q2, q2, 1.0)
    prog.AddQuadraticErrorCost(np.eye(num_q), q1, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(np.eye(num_q), q2, trajopt.control_points()[:, -1])

    # ============= Initial guess =============
    guess_qs = []
    for i in range(num_control_points):
        s = i / (num_control_points - 1)

        if s <= 0.5:
            phase_s = s / 0.5
            guess_q = q1 + phase_s * (q_safe - q1)
        else:
            phase_s = (s - 0.5) / 0.5
            guess_q = q_safe + phase_s * (q2 - q_safe)

        prog.SetInitialGuess(trajopt.control_points()[:, i], guess_q)

    # do same but w 100 points and append to guess_qs
    for i in range(100):
        s = i / 99

        if s <= 0.5:
            phase_s = s / 0.5
            guess_q = q1 + phase_s * (q_safe - q1)
        else:
            phase_s = (s - 0.5) / 0.5
            guess_q = q_safe + phase_s * (q2 - q_safe)

        guess_qs.append(guess_q)

    control_pts_guess = prog.GetInitialGuess(trajopt.control_points())
    initial_spline = BsplineTrajectory(trajopt.basis(), control_pts_guess)

    max_vel = optimization_plant.GetVelocityUpperLimits().flatten()
    if q_safe is not None:
        dist = np.abs(q_safe - q1) + np.abs(q2 - q_safe)
    else:
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

    evaluate_at_s = np.linspace(0, 1, num_samples)
    for s in evaluate_at_s:
        trajopt.AddPathPositionConstraint(collision_constraint, s)

    return trajopt, prog, guess_qs, initial_spline


def solve_kinematic_traj_opt(
    station,
    q1: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray = None,
    acc_limits: np.ndarray = None,
    duration_constraints: tuple[float, float] = (0.5, 10.0),
    num_control_points: int = 10,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    q_safe: np.ndarray = None,
    num_samples: int = 25,
    minimum_distance: float = 0.001,
):
    """
    Set up, solve, and reparameterize a kinematic trajectory from q1 to q2.

    Returns:
        trajectory: reparameterized trajectory (or None on failure)
        success: bool
    """
    if vel_limits is None:
        vel_limits = np.full(7, 1.0)
    if acc_limits is None:
        acc_limits = np.full(7, 1.0)

    (
        trajopt,
        prog,
        guess_qs,
        initial_spline,
    ) = setup_trajectory_optimization_from_q1_to_q2(
        station,
        q1,
        q_safe,
        q2,
        vel_limits,
        acc_limits,
        duration_constraints,
        num_control_points,
        duration_cost,
        path_length_cost,
        num_samples,
        minimum_distance,
    )

    print("Solving trajectory optimization...")
    result = Solve(prog)

    if not result.is_success():
        print(
            colored(
                f"❌ Trajectory optimization failed! Solver: {result.get_solver_id().name()}",
                "red",
            )
        )
        return None, False, guess_qs, initial_spline, None

    print(colored("✓ Trajectory optimization succeeded!", "green"))

    trajectory, toppra_success, solved_spline = resolve_with_toppra(
        station, trajopt, result, vel_limits, acc_limits
    )

    if (not toppra_success) or trajectory is None:
        print(colored("❌ TOPPRA failed to reparameterize the trajectory!", "red"))
        return None, False, guess_qs, initial_spline, solved_spline
    else:
        print(
            colored(
                f"✓ TOPPRA succeeded! Duration: {trajectory.end_time():.2f}s", "green"
            )
        )

    return trajectory, True, guess_qs, initial_spline, solved_spline


def solve_kinematic_traj_opt_async(
    station,
    q1: np.ndarray,
    q_safe: np.ndarray,
    q2: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    result_dict: dict,
    check_final_trajectory=False,  # We'll check for collisions separately after reparameterization
    duration_constraints: tuple[float, float] = (0.5, 10.0),
    num_control_points: int = 10,
    duration_cost: float = 1.0,
    path_length_cost: float = 1.0,
    num_samples: int = 25,
    minimum_distance: float = 0.001,
):
    """
    Wrapper for solve_kinematic_traj_opt intended to be run in a background thread.
    Populates result_dict with 'trajectory', 'success', and 'ready' keys.
    """
    (
        trajectory,
        success,
        guess_qs,
        initial_spline,
        solved_spline,
    ) = solve_kinematic_traj_opt(
        station=station,
        q1=q1,
        q_safe=q_safe,
        q2=q2,
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        duration_constraints=duration_constraints,
        num_control_points=num_control_points,
        duration_cost=duration_cost,
        path_length_cost=path_length_cost,
        num_samples=num_samples,
        minimum_distance=minimum_distance,
    )

    # check if there's collisions going down the generated trajectory
    if success and trajectory is not None and check_final_trajectory:
        opt_plant = station.get_optimization_plant()
        opt_plant_context = station.internal_station.get_optimization_plant_context()
        opt_sg = station.internal_station.get_optimization_diagram_sg()
        opt_sg_context = station.internal_station.get_optimization_diagram_sg_context()

        t_start = trajectory.start_time()
        t_end = trajectory.end_time()
        check_times = np.linspace(t_start, t_end, num_samples)

        collision_detected = False
        for t in check_times:
            q = trajectory.value(t).flatten()
            opt_plant.SetPositions(opt_plant_context, q)
            query_object = opt_sg.get_query_output_port().Eval(opt_sg_context)
            if query_object.HasCollisions():
                collision_detected = True
                break

        if collision_detected:
            print(colored("❌ Collision detected in generated trajectory!", "red"))
            success = False
        else:
            print(colored("✓ No collisions detected in trajectory.", "green"))

    result_dict["trajectory"] = trajectory
    result_dict["success"] = success
    result_dict["guess_qs"] = guess_qs
    result_dict["initial_spline"] = initial_spline
    result_dict["solved_spline"] = solved_spline
    result_dict["ready"] = True
