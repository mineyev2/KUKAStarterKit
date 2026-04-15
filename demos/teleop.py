# General imports
import argparse

from enum import Enum, auto
from pathlib import Path

import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    JointSliders,
    KinematicTrajectoryOptimization,
    MeshcatPoseSliders,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    Simulator,
    Solve,
    TrajectorySource,
)
from pydrake.systems.drawing import plot_system_graphviz
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import (
    add_collision_constraints_to_trajectory,
    resolve_with_toppra,
    setup_trajectory_optimization_from_q1_to_q2,
)
from iiwa_setup.util.visualizations import draw_sphere
from utils.hemisphere_solver import load_joint_poses_from_csv
from utils.kuka_geo_kin import KinematicsSolver


class State(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()


def main(use_hardware: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
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
            scenario=scenario,
            use_hardware=use_hardware,
        ),
    )
    # Load all values I use later
    controller_plant = station.get_iiwa_controller_plant()

    # eef pose sliders
    # Get initial end-effector pose from robot's default joint configuration
    plant_context = controller_plant.CreateDefaultContext()
    controller_plant.SetPositions(
        plant_context,
        controller_plant.GetPositions(plant_context),  # Uses YAML-specified positions
    )

    internal_plant = station.get_internal_plant()
    internal_plant_context = internal_plant.CreateDefaultContext()
    initial_eef_pose = internal_plant.GetFrameByName(
        "microscope_tip_link"
    ).CalcPoseInWorld(internal_plant_context)

    # eef pose sliders
    # Set up teleop widgets
    eef_teleop = builder.AddSystem(
        MeshcatPoseSliders(
            station.internal_meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
            initial_pose=initial_eef_pose,
        )
    )

    # Create dummy constant position source (using station's default position)
    default_position = station.get_iiwa_controller_plant().GetPositions(
        station.get_iiwa_controller_plant().CreateDefaultContext()
    )
    dummy = builder.AddSystem(ConstantVectorSource(default_position))

    # Add connections (using dummy instead of teleop)
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Build diagram
    diagram = builder.Build()

    # ==================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Execute Trajectory")
    station.internal_meshcat.AddButton("Stop Simulation")
    execute_trajectory_clicks = 0

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    vel_limits = np.full(7, 0.3)  # rad/s
    acc_limits = np.full(7, 0.3)  # rad/s^2
    prev_state = None
    state = State.IDLE

    # Create trajectory
    eef_pose_teleop_context = eef_teleop.GetMyContextFromRoot(simulator.get_context())
    eef_pose_prev = RigidTransform(
        eef_teleop.get_output_port().Eval(eef_pose_teleop_context)
    )
    eef_pose_latest = RigidTransform(eef_pose_prev)

    kinematics_solver = KinematicsSolver(station)

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        # Button Management
        new_execute_trajectory_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        if prev_state != state:
            print(colored(f"State changed: {prev_state} -> {state}", "cyan"))
            prev_state = state

        if state == State.IDLE:
            # Check if current update is different from most recent update.
            # If it is, plan traj from current pose to new pose (not latest pose to new pose)
            eef_pose_teleop_context = eef_teleop.GetMyContextFromRoot(
                simulator.get_context()
            )
            eef_pose_current = eef_teleop.get_output_port().Eval(
                eef_pose_teleop_context
            )
            if not eef_pose_current.IsNearlyEqualTo(eef_pose_latest, 1e-6):
                print(
                    colored(
                        "Teleop sliders changed, re-planning trajectory...", "yellow"
                    )
                )

                # Step 1) Solve IK for desired pose
                Q = kinematics_solver.IK_for_microscope_multiple_elbows(
                    eef_pose_current.rotation().matrix(),
                    eef_pose_current.translation(),
                )

                # Step 2) Find IK closest to current joint values
                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )
                q_des = kinematics_solver.find_closest_solution(Q, q_curr)

                # Step 3) Plan trajectory from current joint values to IK solution
                (
                    trajopt,
                    prog,
                    traj_plot_state,
                ) = setup_trajectory_optimization_from_q1_to_q2(
                    station=station,
                    q1=q_curr,
                    q2=q_des,
                    vel_limits=vel_limits,
                    acc_limits=acc_limits,
                    duration_constraints=(0.5, 5.0),
                    num_control_points=10,
                    duration_cost=1.0,
                    path_length_cost=1.0,
                    visualize_solving=True,
                )

                # Solve for initial guess
                traj_plot_state["rgba"] = Rgba(
                    1, 0.5, 0, 1
                )  # Set initial guess color to orange
                result = Solve(prog)

                if not result.is_success():
                    print(colored("Trajectory optimization failed!", "red"))
                    # Reset errors back
                    eef_pose_latest = RigidTransform(eef_pose_current)
                    state = State.IDLE
                    continue
                else:
                    print(colored("✓ Trajectory optimization succeeded!", "green"))

                trajectory = resolve_with_toppra(  # At this point all this is doing is time-optimizing to make the traj as fast as possible
                    station,
                    trajopt,
                    result,
                    vel_limits,
                    acc_limits,
                )
                print(
                    f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s"
                )

                eef_pose_latest = RigidTransform(
                    eef_pose_current
                )  # Update latest pose to current pose after planning new trajectory

            if new_execute_trajectory_clicks > execute_trajectory_clicks:
                eef_pose_prev = RigidTransform(eef_pose_current)
                trajectory_start_time = simulator.get_context().get_time()
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
                state = State.IDLE

        # Reset buttons here in case there are misclicks while the trajectory is executing
        execute_trajectory_clicks = new_execute_trajectory_clicks

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Execute Trajectory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
