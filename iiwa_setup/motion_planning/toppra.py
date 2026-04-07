import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    BsplineTrajectory,
    MultibodyPlant,
    PathParameterizedTrajectory,
    Toppra,
    Trajectory,
)
from termcolor import colored


def reparameterize_with_toppra(
    trajectory: Trajectory,
    plant: MultibodyPlant,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    num_grid_points: int = 1000,
) -> PathParameterizedTrajectory:
    """Reparameterize a trajectory/ path with Toppra.

    Args:
        trajectory (Trajectory): The trajectory on which the TOPPRA problem will be
        solved.
        plant (MultibodyPlant): The robot that will follow the solved trajectory. Used
        for enforcing torque and frame specific constraints.
        velocity_limits (np.ndarray): The velocity limits of shape (N,) where N is the
        number of robot joint joints.
        acceleration_limits (np.ndarray): The acceleration limits of shape (N,) where N
        is the number of robot joint joints.
        num_grid_points (int, optional): The number of uniform points along the path to
        discretize the problem and enforce constraints at.

    Returns:
        PathParameterizedTrajectory: The reparameterized trajectory.
    """

    # Print debug info
    print("start time:", trajectory.start_time())
    print("end time:", trajectory.end_time())

    # Create gridpoints
    gridpoints = np.linspace(
        trajectory.start_time(), trajectory.end_time(), num_grid_points
    )

    # deriv_508 = trajectory.EvalDerivative(gridpoints[508], 1)
    # print(f"q' at knot 508: {deriv_508.flatten()}")
    # print(f"q' norm at knot 508: {np.linalg.norm(deriv_508):.6f}")

    print(f"\nAttempting TOPPRA with {num_grid_points} gridpoints...")
    try:
        toppra = Toppra(
            path=trajectory,
            plant=plant,
            gridpoints=gridpoints,
        )

        toppra.AddJointVelocityLimit(-velocity_limits, velocity_limits)
        toppra.AddJointAccelerationLimit(-acceleration_limits, acceleration_limits)

        time_trajectory = toppra.SolvePathParameterization()
        if time_trajectory is None:
            print(
                colored(
                    "❌ TOPPRA failed: SolvePathParameterization returned None.", "red"
                )
            )
            return None, False
    except Exception as e:
        print(colored(f"❌ TOPPRA failed with error: {e}", "red"))
        return None, False
    print("✓ TOPPRA succeeded!")

    return PathParameterizedTrajectory(trajectory, time_trajectory), True
