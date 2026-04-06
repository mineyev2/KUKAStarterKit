"""
Replays hemisphere_q_solutions.csv step by step using GCS alternate path planning.

Usage:
    python demos/replay_q_solutions.py
    python demos/replay_q_solutions.py --start_idx 5
    python demos/replay_q_solutions.py --csv outputs/my_solutions.csv
"""

import argparse
import threading

from enum import Enum, auto
from pathlib import Path

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import solve_kinematic_traj_opt_async
from utils.planning import move_along_trajectory, plot_trajectory_in_meshcat


class State(Enum):
    IDLE = auto()
    COMPUTING_PATH = auto()
    MOVING = auto()
    DONE = auto()


def load_q_solutions(csv_path: Path):
    """Load CSV, skip NaN rows, return list of (original_row_idx, q) tuples."""
    raw = np.loadtxt(csv_path, delimiter=",")
    valid = []
    for i, row in enumerate(raw):
        if not np.isnan(row).any():
            valid.append((i, row))
    print(
        f"Loaded {len(valid)} valid configs from {csv_path} ({len(raw) - len(valid)} NaN rows skipped)"
    )
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Index into valid q list to start at (0-based after NaN filtering)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(
            Path(__file__).parent.parent / "outputs" / "hemisphere_q_solutions.csv"
        ),
    )
    args = parser.parse_args()

    # ==================================================================
    # Load Q solutions
    # ==================================================================
    q_solutions = load_q_solutions(Path(args.csv))

    if not q_solutions:
        print(colored("No valid q solutions found. Exiting.", "red"))
        return

    start_idx = args.start_idx
    if start_idx >= len(q_solutions):
        print(
            colored(
                f"start_idx {start_idx} out of range ({len(q_solutions)} valid configs). Clamping to 0.",
                "yellow",
            )
        )
        start_idx = 0

    initial_q = q_solutions[start_idx][1]
    print(
        f"Starting at solution index {start_idx} (row {q_solutions[start_idx][0]}): {np.rad2deg(initial_q).round(2)} deg"
    )

    # ==================================================================
    # Scenario / diagram
    # ==================================================================
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

    hemisphere_dist = 0.8
    hemisphere_angle = np.deg2rad(60)
    hemisphere_radius = 0.08

    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2

    builder = DiagramBuilder()
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_dist=hemisphere_dist,
            hemisphere_angle=hemisphere_angle,
            hemisphere_radius=hemisphere_radius,
            use_hardware=False,
        ),
    )

    # Start robot at initial_q
    dummy = builder.AddSystem(ConstantVectorSource(initial_q))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.position"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # ==================================================================
    # Meshcat buttons
    # ==================================================================
    meshcat = station.internal_meshcat
    meshcat.AddButton("Next Config")
    meshcat.AddButton("Stop")

    # ==================================================================
    # State machine
    # ==================================================================
    state = State.IDLE
    curr_solution_idx = start_idx  # index into q_solutions list
    trajectory_start_time = 0.0

    gcs_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "guess_qs": None,
    }
    gcs_thread = None

    num_next_clicks = 0

    print(
        colored(f"\nReady. {len(q_solutions) - start_idx} configs remaining.", "cyan")
    )
    print(
        colored("Press 'Next Config' in Meshcat to step to the next solution.", "cyan")
    )

    while meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)

        if state == State.IDLE:
            if meshcat.GetButtonClicks("Next Config") > num_next_clicks:
                num_next_clicks += 1
                next_idx = curr_solution_idx + 1

                if next_idx >= len(q_solutions):
                    print(colored("✓ All configs visited.", "green"))
                    state = State.DONE
                    continue

                row_idx, q_target = q_solutions[next_idx]
                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )
                _, q_current_safe = q_solutions[curr_solution_idx]

                print(
                    colored(
                        f"\nPlanning: solution {curr_solution_idx} → {next_idx} (row {row_idx})",
                        "cyan",
                    )
                )
                print(f"  target q (deg): {np.rad2deg(q_target).round(2)}")

                gcs_result["ready"] = False
                gcs_result["success"] = False
                gcs_thread = threading.Thread(
                    target=solve_kinematic_traj_opt_async,
                    args=(
                        station,
                        q_current,
                        q_current_safe,
                        q_target,
                        vel_limits,
                        acc_limits,
                        gcs_result,
                    ),
                    daemon=True,
                )
                gcs_thread.start()
                state = State.COMPUTING_PATH

        elif state == State.COMPUTING_PATH:
            if gcs_result["ready"]:
                if gcs_result["success"]:
                    plot_trajectory_in_meshcat(
                        station, gcs_result["trajectory"], name="replay_traj"
                    )
                    trajectory_start_time = simulator.get_context().get_time()
                    state = State.MOVING
                    print(colored("✓ Path planned. Moving...", "green"))
                else:
                    next_idx = curr_solution_idx + 1
                    _, q_target = q_solutions[next_idx]
                    print(
                        colored(
                            f"❌ GCS planning failed for solution {next_idx} (row {q_solutions[next_idx][0]}). Skipping.",
                            "yellow",
                        )
                    )
                    curr_solution_idx = next_idx  # skip it, curr_idx doesn't update
                    state = State.IDLE

        elif state == State.MOVING:
            traj_complete = move_along_trajectory(
                gcs_result["trajectory"],
                trajectory_start_time,
                simulator,
                station,
            )
            if traj_complete:
                curr_solution_idx += 1
                row_idx = q_solutions[curr_solution_idx][0]
                print(
                    colored(
                        f"✓ Arrived at solution {curr_solution_idx} (row {row_idx}). "
                        f"{len(q_solutions) - curr_solution_idx - 1} remaining.",
                        "green",
                    )
                )
                state = State.IDLE

        elif state == State.DONE:
            break

    print("Simulation ended.")


if __name__ == "__main__":
    main()
