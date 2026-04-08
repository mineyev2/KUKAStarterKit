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
from pydrake.geometry import Rgba
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import solve_kinematic_traj_opt_async
from utils.planning import (
    move_along_trajectory,
    plot_configs_in_meshcat,
    plot_trajectory_in_meshcat,
)


class State(Enum):
    IDLE = auto()
    COMPUTING_PATH = auto()
    PREVIEW = auto()
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
    # hemisphere_angle = np.deg2rad(60)
    hemisphere_angle = np.deg2rad(0)
    hemisphere_radius = 0.08

    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2

    # -1.16511234  1.05194255  1.29432601 -2.11578569 -0.26286749 -0.89005845 2.31989321
    q_initial = np.array(
        [
            -1.16511234,
            1.05194255,
            1.29432601,
            -2.11578569,
            -0.26286749,
            -0.89005845,
            2.31989321,
        ]
    )

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
    meshcat.AddButton("Show Initial Guess")
    meshcat.AddButton("Play Solved Path")
    meshcat.AddButton("Stop")

    joint_limits_deg = [170, 120, 170, 120, 170, 120, 175]
    for i in range(7):
        lim = joint_limits_deg[i]
        meshcat.AddSlider(f"J{i+1} (deg)", -lim, lim, 0.1, 0.0)

    # ==================================================================
    # State machine
    # ==================================================================
    state = State.IDLE
    curr_solution_idx = start_idx  # index into q_solutions list
    trajectory_start_time = 0.0

    traj_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "guess_qs": None,
        "initial_spline": None,
        "solved_spline": None,
    }
    traj_thread = None

    num_next_clicks = 0
    num_guess_clicks = 0
    num_solved_clicks = 0
    show_guess_mode = False  # True when triggered by "Show Initial Guess"
    play_solved_mode = False  # True when triggered by "Play Solved Path"

    print(
        colored(f"\nReady. {len(q_solutions) - start_idx} configs remaining.", "cyan")
    )
    print(colored("Press 'Next Config' or 'Show Initial Guess' in Meshcat.", "cyan"))

    while meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)

        station_context = station.GetMyContextFromRoot(simulator.get_context())
        q_now = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
        for i, q in enumerate(q_now):
            meshcat.SetSliderValue(f"J{i+1} (deg)", round(np.rad2deg(q), 1))

        if state == State.IDLE:
            next_clicked = meshcat.GetButtonClicks("Next Config") > num_next_clicks
            guess_clicked = (
                meshcat.GetButtonClicks("Show Initial Guess") > num_guess_clicks
            )
            solved_clicked = (
                meshcat.GetButtonClicks("Play Solved Path") > num_solved_clicks
            )

            if not next_clicked and not guess_clicked and not solved_clicked:
                continue

            # Delete old traj visuals
            meshcat.Delete("initial_spline_traj")
            meshcat.Delete("solved_spline_traj")
            meshcat.Delete("replay_traj")

            if next_clicked:
                num_next_clicks += 1
                show_guess_mode = False
                play_solved_mode = False
            elif guess_clicked:
                num_guess_clicks += 1
                show_guess_mode = True
                play_solved_mode = False
            else:
                num_solved_clicks += 1
                show_guess_mode = False
                play_solved_mode = True

            next_idx = curr_solution_idx + 1

            if next_idx >= len(q_solutions):
                print(colored("✓ All configs visited.", "green"))
                state = State.DONE
                continue

            row_idx, q_target = q_solutions[next_idx]
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )
            # _, q_current_safe = q_solutions[curr_solution_idx]

            print(
                colored(
                    f"\n{'[Guess preview] ' if show_guess_mode else ''}Planning: solution {curr_solution_idx} → {next_idx} (row {row_idx})",
                    "cyan",
                )
            )
            print(f"  target q (deg): {np.rad2deg(q_target).round(2)}")

            # check_final_trajectory=False,  # We'll check for collisions separately after reparameterization
            # duration_constraints: tuple[float, float] = (0.5, 10.0),
            # num_control_points: int = 10,
            # duration_cost: float = 1.0,
            # path_length_cost: float = 1.0,
            # num_samples: int = 25,
            # minimum_distance: float = 0.001,

            traj_result["ready"] = False
            traj_result["success"] = False
            traj_thread = threading.Thread(
                target=solve_kinematic_traj_opt_async,
                args=(
                    station,
                    q_current,
                    q_initial,
                    q_target,
                    vel_limits,
                    acc_limits,
                    traj_result,
                    False,
                    (0.5, 10.0),
                    10,
                ),
                daemon=True,
            )
            traj_thread.start()
            state = State.COMPUTING_PATH

        elif state == State.COMPUTING_PATH:
            if traj_result["ready"]:
                plot_trajectory_in_meshcat(
                    station,
                    traj_result["initial_spline"],
                    name="initial_spline_traj",
                    rgba=Rgba(1, 1, 0, 1),
                )
                if show_guess_mode:
                    spline = traj_result["initial_spline"]
                    if spline is not None:
                        ts = np.linspace(spline.start_time(), spline.end_time(), 50)
                        qs = [spline.value(t).flatten() for t in ts]
                        for q in qs + list(reversed(qs)):
                            station.GetInputPort("iiwa.position").FixValue(
                                station_context, q
                            )
                            simulator.AdvanceTo(
                                simulator.get_context().get_time() + 0.1
                            )
                            for i, qi in enumerate(q):
                                meshcat.SetSliderValue(
                                    f"J{i+1} (deg)", round(np.rad2deg(qi), 1)
                                )
                    print(colored("✓ Guess preview done.", "cyan"))
                    state = State.IDLE
                elif play_solved_mode:
                    spline = traj_result["solved_spline"]
                    if spline is not None:
                        plot_trajectory_in_meshcat(
                            station,
                            spline,
                            name="solved_spline_traj",
                            rgba=Rgba(0, 1, 1, 1),
                        )
                        ts = np.linspace(spline.start_time(), spline.end_time(), 50)
                        qs = [spline.value(t).flatten() for t in ts]
                        for q in qs + list(reversed(qs)):
                            station.GetInputPort("iiwa.position").FixValue(
                                station_context, q
                            )
                            simulator.AdvanceTo(
                                simulator.get_context().get_time() + 0.1
                            )
                            for i, qi in enumerate(q):
                                meshcat.SetSliderValue(
                                    f"J{i+1} (deg)", round(np.rad2deg(qi), 1)
                                )
                    else:
                        print(
                            colored("❌ No solved spline available for preview!", "red")
                        )
                    print(colored("✓ Solved path preview done.", "cyan"))
                    state = State.IDLE
                elif traj_result["success"]:
                    plot_trajectory_in_meshcat(
                        station,
                        traj_result["solved_spline"],
                        name="solved_spline_traj",
                        rgba=Rgba(0, 1, 1, 1),
                    )
                    plot_trajectory_in_meshcat(
                        station, traj_result["trajectory"], name="replay_traj"
                    )
                    trajectory_start_time = simulator.get_context().get_time()
                    print(
                        f"traj start q: {np.rad2deg(traj_result['trajectory'].value(0).flatten()).round(1)}"
                    )
                    print(f"robot actual q: {np.rad2deg(q_current).round(1)}")
                    print(colored("✓ Path planned. Moving...", "green"))
                    state = State.MOVING
                else:
                    next_idx = curr_solution_idx + 1
                    _, q_target = q_solutions[next_idx]
                    print(
                        colored(
                            f"❌ Trajectory planning failed for solution {next_idx} (row {q_solutions[next_idx][0]}). Skipping.",
                            "yellow",
                        )
                    )
                    quit()
                    # curr_solution_idx = next_idx  # skip it, curr_idx doesn't update
                    # state = State.IDLE

        elif state == State.MOVING:
            traj_complete = move_along_trajectory(
                traj_result["trajectory"],
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
