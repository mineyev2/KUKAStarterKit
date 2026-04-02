import argparse

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    BsplineBasis,
    BsplineTrajectory,
    CoulombFriction,
    DiagramBuilder,
    InverseKinematics,
    JointSliders,
    KinematicTrajectoryOptimization,
    KnotVectorType,
    Meshcat,
    MeshcatVisualizer,
    MinimumDistanceLowerBoundConstraint,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    SceneGraphCollisionChecker,
    Simulator,
    Solve,
    SpatialInertia,
    Sphere,
    UnitInertia,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter
from termcolor import colored

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


def draw_sphere(meshcat, name, position, radius=0.01):
    rgba = Rgba(0.0, 1.0, 0.1, 0.5)

    meshcat.SetObject(
        name,
        Sphere(radius),
        rgba,
    )
    meshcat.SetTransform(
        name,
        RigidTransform(np.array(position)),
    )


def main(use_hardware: bool, has_wsg: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: world
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

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=has_wsg, use_hardware=use_hardware
        ),
    )

    # Load teleop sliders
    controller_plant = station.get_iiwa_controller_plant()
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    # Add connections
    builder.Connect(
        teleop.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
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
    station.internal_meshcat.AddButton("Plan Trajectory")
    station.internal_meshcat.AddButton("Move to Goal")

    # ====================================================================
    # Trajectory Optimization + TOPPRA Loop
    # ====================================================================

    # Define goal position and limits
    q_goal = np.array([0, np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
    vel_limits = np.full(7, 0.2)  # rad/s
    acc_limits = np.full(7, 0.2)  # rad/s^2

    # Get plants
    optimization_plant = station.internal_station.get_optimization_plant()
    internal_plant = station.get_internal_plant()

    # Optimization params
    num_q = optimization_plant.num_positions()
    num_control_points = 10

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    move_clicks = 0
    plan_clicks = 0
    path_counter = 0
    trajectory = None
    trajectory_start_time = 0.0
    execute_trajectory = False
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        new_move_clicks = station.internal_meshcat.GetButtonClicks("Move to Goal")
        new_plan_clicks = station.internal_meshcat.GetButtonClicks("Plan Trajectory")
        if new_plan_clicks > plan_clicks:  # Triggered when Plan Trajectory is pressed
            plan_clicks = new_plan_clicks
            print("Planning trajectory to goal...")

            # Get contexts (NOTE: Must do here to have up-to-date values)
            internal_context = station.get_internal_plant_context()
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            optimization_plant_context = (
                station.internal_station.get_optimization_plant_context()
            )
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            trajopt = KinematicTrajectoryOptimization(
                num_q, num_control_points, spline_order=4
            )
            prog = trajopt.get_mutable_prog()

            # ============= Costs =============
            trajopt.AddDurationCost(1.0)
            trajopt.AddPathLengthCost(1.0)

            # ============= Bounds =============
            trajopt.AddPositionBounds(
                optimization_plant.GetPositionLowerLimits(),
                optimization_plant.GetPositionUpperLimits(),
            )
            trajopt.AddVelocityBounds(
                optimization_plant.GetVelocityLowerLimits(),
                optimization_plant.GetVelocityUpperLimits(),
            )

            # ============= Constraints =============
            trajopt.AddDurationConstraint(0.5, 5)  # TODO: May need to adjust later

            # Position
            trajopt.AddPathPositionConstraint(q_current, q_current, 0.0)
            trajopt.AddPathPositionConstraint(q_goal, q_goal, 1.0)
            # Use quadratic consts to encourage q current and q goal
            prog.AddQuadraticErrorCost(
                np.eye(num_q), q_current, trajopt.control_points()[:, 0]
            )
            prog.AddQuadraticErrorCost(
                np.eye(num_q), q_goal, trajopt.control_points()[:, -1]
            )

            # Velocity (TOPPRA assumes zero start and end velocities)
            trajopt.AddPathVelocityConstraint(
                np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
            )
            trajopt.AddPathVelocityConstraint(
                np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
            )

            def PlotPath(control_points):
                """
                Visualize the end-effector path in Meshcat
                """
                rgba = Rgba(0, 1, 0, 1)
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
                        internal_plant.GetBodyByName("iiwa_link_7"),
                    )
                    ee_positions.append(X_WB.translation())
                ee_positions = np.array(ee_positions).T  # shape (3, N)
                station.internal_meshcat.SetLine(
                    f"positions_path_{path_counter}",
                    ee_positions,
                    line_width=0.05,
                    rgba=rgba,
                )

            prog.AddVisualizationCallback(
                PlotPath, trajopt.control_points().reshape((-1,))
            )

            # Solve for initial guess
            result = Solve(prog)
            if not result.is_success():
                print("Trajectory optimization failed, even without collisions!")
                print(result.get_solver_id().name())
            trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

            # Add collision constraints for next solve
            collision_constraint = MinimumDistanceLowerBoundConstraint(
                optimization_plant,
                0.001,
                optimization_plant_context,
                None,
            )
            evaluate_at_s = np.linspace(0, 1, 25)  # TODO: Use a diff value?
            for s in evaluate_at_s:
                trajopt.AddPathPositionConstraint(collision_constraint, s)

            # Solve for trajectory with collision avoidance
            result = Solve(prog)
            if not result.is_success():
                print(
                    colored(
                        "Trajectory optimization with collision avoidance failed!",
                        "red",
                    )
                )
                print(result.get_solver_id().name())
                continue

            print("Trajectory optimization succeeded!")

            # Reparameterize with TOPPRA
            geometric_path = trajopt.ReconstructTrajectory(result)

            # Plot joint trajectories
            ts = np.linspace(
                geometric_path.start_time(), geometric_path.end_time(), 100
            )
            qs = np.array([geometric_path.value(t) for t in ts])
            plt.figure()
            for i in range(qs.shape[1]):
                plt.plot(ts, qs[:, i], label=f"Joint {i+1}")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint Position [rad]")
            plt.title("Geometric Path Joint Positions")
            plt.legend()
            plt.savefig("output/geometric_path.png")
            plt.close()

            trajectory = reparameterize_with_toppra(
                geometric_path,
                controller_plant,
                velocity_limits=vel_limits,
                acceleration_limits=acc_limits,
            )

            print(
                f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s"
            )

        # If we have a trajectory, execute it
        if new_move_clicks > move_clicks:  # Triggered when Move to Goal is pressed
            move_clicks = new_move_clicks
            if trajectory is None:
                print("No trajectory planned yet!")
            else:
                print("Executing trajectory...")
                execute_trajectory = True
                trajectory_start_time = simulator.get_context().get_time()

        if execute_trajectory:
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
                print("✓ Trajectory execution complete!")
                trajectory = None
                execute_trajectory = False

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Trajectory")
    station.internal_meshcat.DeleteButton("Move to Goal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--has_wsg",
        action="store_true",
        help="Whether the iiwa has a WSG gripper or not.",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware, has_wsg=args.has_wsg)
