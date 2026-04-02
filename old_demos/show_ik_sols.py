import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    AddFrameTriadIllustration,
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

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import compute_simple_traj_from_q1_to_q2
from iiwa_setup.util.visualizations import draw_sphere

# Personal files
from utils.hemisphere_solver import generate_hemisphere_joint_poses
from utils.kuka_geo_kin import KinematicsSolver


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
        IiwaHardwareStationDiagram(scenario=scenario, use_hardware=use_hardware),
    )

    # Load all values I use later
    internal_station = station.internal_station
    internal_plant = station.get_internal_plant()
    controller_plant = station.get_iiwa_controller_plant()

    # Frames
    tip_frame = internal_plant.GetFrameByName("microscope_tip_link")
    link7_frame = internal_plant.GetFrameByName("iiwa_link_7")

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

    # Add coordinate frames
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=internal_plant,
        frame=tip_frame,
        length=0.05,
        radius=0.002,
        name="microscope_tip_frame",
    )

    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=internal_plant,
        frame=link7_frame,
        length=0.1,
        radius=0.002,
        name="iiwa_link_7_frame",
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
    # Compute all joint poses for sphere scanning
    # ====================================================================
    kinematics_solver = KinematicsSolver(station)

    # Solve example IK
    target_rot = np.eye(3)
    target_pos = np.array([0.7, 0.0, 0.6])
    vel_limits = np.full(7, 1.5)  # rad/s
    acc_limits = np.full(7, 1.5)  # rad/s²

    draw_sphere(
        station.internal_meshcat,
        "target_sphere",
        position=target_pos,
        radius=0.02,
    )

    q_sols = kinematics_solver.IK_for_microscope(
        target_rot,
        target_pos,
        psi=0,
    )

    print("IK solutions for test pose:")
    for idx, q_sol in enumerate(q_sols):
        print(f"Solution {idx + 1}: {q_sol}")

    controller_plant = station.get_iiwa_controller_plant()

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    move_clicks = 0
    ik_idx = 0
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if station.internal_meshcat.GetButtonClicks("Move to Goal") > move_clicks:
            move_clicks = station.internal_meshcat.GetButtonClicks("Move to Goal")

            if ik_idx >= len(q_sols):
                print("All IK solutions have been executed.")
                continue

            q_goal = q_sols[ik_idx]
            print(f"Moving to goal: {q_goal}")

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            # Read the measured position from the station (works for both Sim and Hardware)
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            traj = compute_simple_traj_from_q1_to_q2(
                controller_plant,
                q_current,
                q_goal,
                vel_limits=vel_limits,
                acc_limits=acc_limits,
            )

            t_traj = 0.0
            dt = 0.01
            t_start = simulator.get_context().get_time()

            while t_traj < traj.end_time():
                q_d = traj.value(t_traj).flatten()
                teleop.SetPositions(q_d)

                step = min(dt, traj.end_time() - t_traj)
                simulator.AdvanceTo(t_start + t_traj + step)
                t_traj += step

            # Print microscope tip position after reaching goal
            station_context = station.get_internal_plant_context()
            X_W_TIP = station.get_internal_plant().CalcRelativeTransform(
                station_context,
                station.get_internal_plant().world_frame(),
                station.get_internal_plant().GetFrameByName("microscope_tip_link"),
            )
            tip_pos = X_W_TIP.translation()
            print(
                f"Reached IK solution {ik_idx + 1}. Microscope tip position: {tip_pos}"
            )

            ik_idx += 1

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

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
