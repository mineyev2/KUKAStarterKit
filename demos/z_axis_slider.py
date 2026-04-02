"""
z_axis_slider.py

State machine:
  IDLE                  → press "Plan Move to Start"
  COMPUTING_MOVE_TO_START → background GCS thread running
  PLANNING_MOVE_TO_START  → trajectory ready, press "Move to Start" to execute
  MOVING_TO_START       → executing trajectory to default_position
  AT_START              → robot at start pose; Z Offset slider is now live

Run with:
    python z_axis_slider.py               # simulation
    python z_axis_slider.py --use_hardware  # real hardware
"""

import argparse
import threading

from enum import Enum, auto

import numpy as np

from demo_config import get_config
from manipulation.station import LoadScenario
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    RigidTransform,
    Simulator,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import solve_kinematic_traj_opt_async
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import (
    generate_hemisphere_waypoints,
    plot_configs_in_meshcat,
    plot_trajectory_in_meshcat,
)


class State(Enum):
    IDLE = auto()
    COMPUTING_MOVE_TO_START = auto()
    PLANNING_MOVE_TO_START = auto()
    MOVING_TO_START = auto()
    AT_START = auto()


def main(use_hardware: bool) -> None:
    cfg = get_config(use_hardware)

    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    plant_config:
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

    # ------------------------------------------------------------------
    # Parameters  (identical to scan_object_and_save_frames2.py)
    # ------------------------------------------------------------------
    hemisphere_dist = 0.8
    hemisphere_angle = np.deg2rad(60)
    hemisphere_radius = 0.100
    hemisphere_pos = np.array(
        [
            hemisphere_dist * np.cos(hemisphere_angle),
            hemisphere_dist * np.sin(hemisphere_angle),
            0.36,
        ]
    )
    hemisphere_axis = np.array(
        [-np.cos(hemisphere_angle), -np.sin(hemisphere_angle), 0]
    )

    elbow_angle = np.deg2rad(135)

    r = np.array([0, 0, -1])
    v = np.array([0, 1, 0])

    # Same initial hold position as scan_object_and_save_frames2.py
    default_position = np.array([hemisphere_angle, 0.1, 0, -1.2, 0, 1.6, 0])

    # Hemisphere waypoints — move-to-start targets hemisphere_waypoints[0]
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=50,
        coverage=0.40,
    )

    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2

    # ------------------------------------------------------------------
    # Build diagram
    # ------------------------------------------------------------------
    builder = DiagramBuilder()

    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_dist=hemisphere_dist,
            hemisphere_angle=hemisphere_angle,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    kinematics_solver = KinematicsSolver(station, r, v)

    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=camera_frame,
        length=0.1,
        radius=0.002,
        name="camera_link",
    )

    diagram = builder.Build()

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    # ------------------------------------------------------------------
    # hemisphere_waypoints[0] is both the move-to-start target and the
    # z-slider reference pose (same as scan_object_and_save_frames2.py)
    # ------------------------------------------------------------------
    start_pose: RigidTransform = hemisphere_waypoints[0]

    print(
        colored(
            f"✓ Start pose (hemisphere_waypoints[0]): {start_pose.translation()}",
            "cyan",
        )
    )

    # ------------------------------------------------------------------
    # Meshcat UI
    # ------------------------------------------------------------------
    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Plan Move to Start")
    station.internal_meshcat.AddButton("Move to Start")

    # Z-offset slider — only active once AT_START; ±50 mm, 0.5 mm step
    station.internal_meshcat.AddSlider("Z Offset (mm)", -50.0, 50.0, 0.5, 0.0)

    for i in range(7):
        station.internal_meshcat.AddSlider(
            f"Joint {i+1} (deg)", -180.0, 180.0, 0.1, 0.0
        )

    # ------------------------------------------------------------------
    # State machine setup
    # ------------------------------------------------------------------
    state = State.IDLE
    prev_state = None

    move_to_start_gcs_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "guess_qs": None,
    }

    initial_trajectory = None
    trajectory_start_time = 0.0

    last_z_offset_mm = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state != prev_state:
            print(colored(f"State: {prev_state} → {state}", "grey"))
            prev_state = state

        station_context = station.GetMyContextFromRoot(simulator.get_context())

        # ---- Update joint display sliders --------------------------------
        q_current = station.GetOutputPort("iiwa.position_measured").Eval(
            station_context
        )
        for i in range(7):
            station.internal_meshcat.SetSliderValue(
                f"Joint {i+1} (deg)", np.rad2deg(q_current[i])
            )

        # ==================================================================
        if state == State.IDLE:
            if station.internal_meshcat.GetButtonClicks("Plan Move to Start") > 0:
                print(colored("Planning move to start (async)…", "cyan"))

                q_initial = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )

                # IK for hemisphere_waypoints[0] — same logic as scan_object_and_save_frames2.py
                Q = kinematics_solver.IK_for_microscope(
                    start_pose.rotation().matrix(),
                    start_pose.translation(),
                    psi=elbow_angle,
                )
                q_des = kinematics_solver.find_closest_solution(Q, q_initial)

                move_to_start_gcs_result["ready"] = False
                move_to_start_gcs_thread = threading.Thread(
                    target=solve_kinematic_traj_opt_async,
                    args=(
                        station,
                        q_initial,
                        q_initial,
                        q_des,
                        vel_limits,
                        acc_limits,
                        move_to_start_gcs_result,
                    ),
                    daemon=True,
                )
                move_to_start_gcs_thread.start()
                state = State.COMPUTING_MOVE_TO_START

        # ==================================================================
        elif state == State.COMPUTING_MOVE_TO_START:
            if move_to_start_gcs_result["ready"]:
                if move_to_start_gcs_result["success"]:
                    initial_trajectory = move_to_start_gcs_result["trajectory"]

                    plot_configs_in_meshcat(
                        station,
                        move_to_start_gcs_result["guess_qs"],
                        name="guess_traj",
                    )
                    plot_trajectory_in_meshcat(
                        station,
                        initial_trajectory,
                        name="final_traj",
                    )

                    print(
                        colored(
                            "✓ GCS planning complete — press 'Move to Start' to execute.",
                            "green",
                        )
                    )
                    state = State.PLANNING_MOVE_TO_START
                else:
                    print(colored("❌ GCS planning failed!", "red"))
                    state = State.IDLE

        # ==================================================================
        elif state == State.PLANNING_MOVE_TO_START:
            if station.internal_meshcat.GetButtonClicks("Move to Start") > 0:
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_START

        # ==================================================================
        elif state == State.MOVING_TO_START:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= initial_trajectory.end_time():
                q_desired = initial_trajectory.value(traj_time)
                mutable_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    mutable_context, q_desired
                )
            else:
                print(
                    colored(
                        "✓ Arrived at start pose — Z Offset slider is now live.",
                        "green",
                    )
                )
                state = State.AT_START

        # ==================================================================
        elif state == State.AT_START:
            z_offset_mm = station.internal_meshcat.GetSliderValue("Z Offset (mm)")

            if abs(z_offset_mm - last_z_offset_mm) > 1e-6:
                last_z_offset_mm = z_offset_mm
                z_offset_m = z_offset_mm / 1000.0

                # Shift start_pose along its own z-axis by z_offset_m
                delta = RigidTransform(np.array([0.0, 0.0, z_offset_m]))
                target_pose = start_pose @ delta

                R = target_pose.rotation().matrix()
                p = target_pose.translation()

                Q = kinematics_solver.IK_for_microscope(R, p, psi=elbow_angle)

                if Q is not None and len(Q) > 0:
                    mutable_context = station.GetMyMutableContextFromRoot(
                        simulator.get_mutable_context()
                    )
                    q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                        station_context
                    )
                    q_des = kinematics_solver.find_closest_solution(Q, q_curr)

                    station.GetInputPort("iiwa.position").FixValue(
                        mutable_context, q_des
                    )
                    print(
                        colored(
                            f"  z={z_offset_mm:+.1f} mm → q={np.round(np.rad2deg(q_des), 1)} deg",
                            "green",
                        )
                    )
                else:
                    print(
                        colored(
                            f"  ⚠ No IK solution at z={z_offset_mm:+.1f} mm — holding position",
                            "yellow",
                        )
                    )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Move to Start")
    station.internal_meshcat.DeleteButton("Move to Start")
    station.internal_meshcat.DeleteSlider("Z Offset (mm)")
    for i in range(7):
        station.internal_meshcat.DeleteSlider(f"Joint {i+1} (deg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microscope tip z-axis slider control")
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Use real KUKA hardware instead of simulation.",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
