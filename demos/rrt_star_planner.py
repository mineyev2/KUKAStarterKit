"""
demos/rrt_star_planner.py

Given two joint configurations q1 and q2 (defined in code), plans an
RRT*-Connect trajectory between them.

Buttons:
  "Solve RRT* Path"  — runs RRT* in a background thread; displays the raw
                       path and smoothed trajectory in Meshcat when done.
  "Execute Path"     — executes the smoothed trajectory on the robot.
  "Stop Simulation"  — exits.

Usage:
    python demos/rrt_star_planner.py
    python demos/rrt_star_planner.py --use_hardware
"""

import argparse
import threading
from enum import Enum, auto

import matplotlib
matplotlib.use("Agg")
import numpy as np

from demo_config import get_config
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
)
from pydrake.systems.primitives import VectorLogSink
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from utils.planning import move_along_trajectory, plot_trajectory_in_meshcat
from utils.RRT import plot_rrt_raw_path_in_meshcat
from utils.RRTStar import plan_rrt_star_async


# ===========================================================================
# Joint configurations (radians)
# ===========================================================================
Q1 = np.deg2rad([-32.06, 56.57, 47.46, -115.28, -0.89, -70.31, -37.64])
Q2 = np.deg2rad([30.0,   30.0,  0.0,   -90.0,    0.0,  -60.0,   0.0])


class State(Enum):
    IDLE = auto()
    PLANNING = auto()
    READY_TO_EXECUTE = auto()
    EXECUTING = auto()
    DONE = auto()


def main(use_hardware: bool) -> None:
    cfg = get_config(use_hardware)
    vel_limits = cfg["vel_limits"]
    acc_limits = cfg["acc_limits"]

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

    # -----------------------------------------------------------------------
    # Build diagram
    # -----------------------------------------------------------------------
    builder = DiagramBuilder()
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            use_hardware=use_hardware,
        ),
    )

    state_logger = builder.AddSystem(VectorLogSink(7))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        state_logger.get_input_port(),
    )

    dummy = builder.AddSystem(ConstantVectorSource(Q1))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.position"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    meshcat = station.internal_meshcat

    # -----------------------------------------------------------------------
    # Joint sliders (read-only display)
    # -----------------------------------------------------------------------
    plant = station.get_internal_plant()
    lower = plant.GetPositionLowerLimits()
    upper = plant.GetPositionUpperLimits()
    for i in range(7):
        meshcat.AddSlider(
            f"Joint {i+1} (deg)",
            np.rad2deg(lower[i]),
            np.rad2deg(upper[i]),
            0.1,
            np.rad2deg(Q1[i]),
        )

    # -----------------------------------------------------------------------
    # Buttons
    # -----------------------------------------------------------------------
    meshcat.AddButton("Solve RRT* Path")
    meshcat.AddButton("Execute Path")
    meshcat.AddButton("Stop Simulation")

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------
    app_state = State.IDLE
    prev_state = None

    rrt_result = {"ready": False, "success": False, "trajectory": None, "path": None}
    trajectory_start_time = 0.0

    num_solve_clicks = 0
    num_execute_clicks = 0

    print(colored("Ready. Open Meshcat and press 'Solve RRT* Path'.", "cyan"))
    print(colored(f"  q1 = {np.rad2deg(Q1).round(2)} deg", "cyan"))
    print(colored(f"  q2 = {np.rad2deg(Q2).round(2)} deg", "cyan"))

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if app_state != prev_state:
            print(colored(f"  [{app_state.name}]", "grey"))
            prev_state = app_state

        # Update joint sliders
        station_context = station.GetMyContextFromRoot(simulator.get_context())
        q_now = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
        for i in range(7):
            meshcat.SetSliderValue(f"Joint {i+1} (deg)", round(np.rad2deg(q_now[i]), 1))

        # ------------------------------------------------------------------
        if app_state == State.IDLE:
            if meshcat.GetButtonClicks("Solve RRT* Path") > num_solve_clicks:
                num_solve_clicks += 1
                rrt_result["ready"] = False
                rrt_result["success"] = False
                rrt_result["trajectory"] = None
                rrt_result["path"] = None

                print(colored("Planning RRT* path…", "cyan"))
                threading.Thread(
                    target=plan_rrt_star_async,
                    args=(station, Q1, Q2, vel_limits, acc_limits, rrt_result),
                    daemon=True,
                ).start()
                app_state = State.PLANNING

        # ------------------------------------------------------------------
        elif app_state == State.PLANNING:
            if rrt_result["ready"]:
                if not rrt_result["success"]:
                    print(colored("❌ RRT* failed to find a path. Press 'Solve RRT* Path' to retry.", "red"))
                    app_state = State.IDLE
                else:
                    print(colored("✓ RRT* succeeded. Displaying path in Meshcat.", "green"))

                    if rrt_result["path"] is not None:
                        meshcat.Delete("rrt_raw_path")
                        plot_rrt_raw_path_in_meshcat(
                            station,
                            rrt_result["path"],
                            name="rrt_raw_path",
                        )

                    if rrt_result["trajectory"] is not None:
                        meshcat.Delete("rrt_traj")
                        plot_trajectory_in_meshcat(
                            station,
                            rrt_result["trajectory"],
                            name="rrt_traj",
                        )

                    print(colored("Press 'Execute Path' to run the trajectory.", "cyan"))
                    app_state = State.READY_TO_EXECUTE

        # ------------------------------------------------------------------
        elif app_state == State.READY_TO_EXECUTE:
            if meshcat.GetButtonClicks("Solve RRT* Path") > num_solve_clicks:
                num_solve_clicks += 1
                rrt_result["ready"] = False
                rrt_result["success"] = False
                print(colored("Re-planning RRT* path…", "cyan"))
                threading.Thread(
                    target=plan_rrt_star_async,
                    args=(station, Q1, Q2, vel_limits, acc_limits, rrt_result),
                    daemon=True,
                ).start()
                app_state = State.PLANNING
            elif meshcat.GetButtonClicks("Execute Path") > num_execute_clicks:
                num_execute_clicks += 1
                trajectory_start_time = simulator.get_context().get_time()
                print(colored("Executing trajectory…", "cyan"))
                app_state = State.EXECUTING

        # ------------------------------------------------------------------
        elif app_state == State.EXECUTING:
            done = move_along_trajectory(
                rrt_result["trajectory"],
                trajectory_start_time,
                simulator,
                station,
            )
            if done:
                print(colored("✓ Trajectory complete.", "green"))
                app_state = State.DONE

        # ------------------------------------------------------------------
        elif app_state == State.DONE:
            if meshcat.GetButtonClicks("Solve RRT* Path") > num_solve_clicks:
                num_solve_clicks += 1
                rrt_result["ready"] = False
                print(colored("Planning new RRT* path…", "cyan"))
                threading.Thread(
                    target=plan_rrt_star_async,
                    args=(station, Q1, Q2, vel_limits, acc_limits, rrt_result),
                    daemon=True,
                ).start()
                app_state = State.PLANNING

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    print(colored("Simulation stopped.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRT* point-to-point planner demo")
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        default=False,
        help="Connect to real iiwa hardware via LCM",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
