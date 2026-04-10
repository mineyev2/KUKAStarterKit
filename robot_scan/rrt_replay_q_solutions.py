"""
Replays hemisphere_q_solutions.csv step by step using RRT-Connect or RRT*-Connect.

Usage:
    python demos/rrt_replay_q_solutions.py
    python demos/rrt_replay_q_solutions.py --planner rrt_star
    python demos/rrt_replay_q_solutions.py --start_idx 5
    python demos/rrt_replay_q_solutions.py --csv outputs/my_solutions.csv
    python demos/rrt_replay_q_solutions.py --step_size 0.05 --max_iter 10000
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
from utils.planning import move_along_trajectory, plot_trajectory_in_meshcat
from utils.RRT import plan_rrt_async, plot_rrt_raw_path_in_meshcat
from utils.RRTStar import plan_rrt_star_async


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
        f"Loaded {len(valid)} valid configs from {csv_path} "
        f"({len(raw) - len(valid)} NaN rows skipped)"
    )
    return valid


def _animate_configs(configs, station, station_context, simulator, meshcat):
    """
    Step the simulated robot through a list of joint configs (forward then reverse).
    Updates joint sliders as it goes.
    """
    for q in list(configs) + list(reversed(configs)):
        station.GetInputPort("iiwa.position").FixValue(station_context, q)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
        for i, qi in enumerate(q):
            meshcat.SetSliderValue(f"J{i+1} (deg)", round(np.rad2deg(qi), 1))


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
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.05,
        help="RRT-Connect step size in joint space (rad). Smaller = safer but slower.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10000,
        help="Maximum RRT iterations per segment.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="rrt",
        choices=["rrt", "rrt_star"],
        help="Planner to use: 'rrt' (RRT-Connect) or 'rrt_star' (RRT*-Connect).",
    )
    parser.add_argument(
        "--rewire_radius",
        type=float,
        default=0.3,
        help="Max rewire radius for RRT* (rad). Only used when --planner rrt_star.",
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
        f"Starting at solution index {start_idx} (row {q_solutions[start_idx][0]}): "
        f"{np.rad2deg(initial_q).round(2)} deg"
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
    hemisphere_angle = np.deg2rad(0)
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
    # Meshcat buttons / sliders
    # ==================================================================
    meshcat = station.internal_meshcat
    meshcat.AddButton("Next Config")
    meshcat.AddButton("Preview Raw Path")
    meshcat.AddButton("Preview Smooth Path")
    meshcat.AddButton("Stop")

    joint_limits_deg = [170, 120, 170, 120, 170, 120, 175]
    for i in range(7):
        lim = joint_limits_deg[i]
        meshcat.AddSlider(f"J{i+1} (deg)", -lim, lim, 0.1, 0.0)

    # ==================================================================
    # State machine
    # ==================================================================
    state = State.IDLE
    curr_solution_idx = start_idx
    trajectory_start_time = 0.0

    traj_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
        "path": None,
    }
    traj_thread = None

    num_next_clicks = 0
    num_raw_clicks = 0
    num_smooth_clicks = 0

    # Flags that determine what happens after planning completes
    preview_raw_mode = False
    preview_smooth_mode = False

    print(
        colored(f"\nReady. {len(q_solutions) - start_idx} configs remaining.", "cyan")
    )
    print(colored("Buttons:", "cyan"))
    print(colored("  'Next Config'        — plan and execute RRT path", "cyan"))
    print(
        colored(
            "  'Preview Raw Path'   — plan and animate raw RRT waypoints (no commit)",
            "cyan",
        )
    )
    print(
        colored(
            "  'Preview Smooth Path'— plan and animate TOPPRA trajectory (no commit)",
            "cyan",
        )
    )
    planner_label = "RRT*-Connect" if args.planner == "rrt_star" else "RRT-Connect"
    rrt_star_info = (
        f", rewire_radius={args.rewire_radius}" if args.planner == "rrt_star" else ""
    )
    print(
        colored(
            f"Planner: {planner_label} | step_size={args.step_size}, max_iter={args.max_iter}{rrt_star_info}",
            "cyan",
        )
    )

    while meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)

        station_context = station.GetMyContextFromRoot(simulator.get_context())
        q_now = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
        for i, q in enumerate(q_now):
            meshcat.SetSliderValue(f"J{i+1} (deg)", round(np.rad2deg(q), 1))

        # ------------------------------------------------------------------
        if state == State.IDLE:
            next_clicked = meshcat.GetButtonClicks("Next Config") > num_next_clicks
            raw_clicked = meshcat.GetButtonClicks("Preview Raw Path") > num_raw_clicks
            smooth_clicked = (
                meshcat.GetButtonClicks("Preview Smooth Path") > num_smooth_clicks
            )

            if not next_clicked and not raw_clicked and not smooth_clicked:
                continue

            # Track clicks and set mode flags
            if next_clicked:
                num_next_clicks += 1
                preview_raw_mode = False
                preview_smooth_mode = False
            elif raw_clicked:
                num_raw_clicks += 1
                preview_raw_mode = True
                preview_smooth_mode = False
            else:
                num_smooth_clicks += 1
                preview_raw_mode = False
                preview_smooth_mode = True

            # Clear old trajectory visuals
            meshcat.Delete("rrt_raw_path")
            meshcat.Delete("rrt_traj")

            next_idx = curr_solution_idx + 1
            if next_idx >= len(q_solutions):
                print(colored("✓ All configs visited.", "green"))
                state = State.DONE
                continue

            row_idx, q_target = q_solutions[next_idx]
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            mode_label = (
                "[Preview Raw] "
                if preview_raw_mode
                else "[Preview Smooth] "
                if preview_smooth_mode
                else ""
            )
            planner_name = (
                "RRT*-Connect" if args.planner == "rrt_star" else "RRT-Connect"
            )
            print(
                colored(
                    f"\n{mode_label}{planner_name} planning: "
                    f"solution {curr_solution_idx} → {next_idx} (row {row_idx})",
                    "cyan",
                )
            )
            print(f"  q_start (deg): {np.rad2deg(q_current).round(2)}")
            print(f"  q_goal  (deg): {np.rad2deg(q_target).round(2)}")

            traj_result["ready"] = False
            traj_result["success"] = False

            if args.planner == "rrt_star":
                target_fn = plan_rrt_star_async
                thread_args = (
                    station,
                    q_current,
                    q_target,
                    vel_limits,
                    acc_limits,
                    traj_result,
                    args.step_size,
                    args.max_iter,
                    args.rewire_radius,
                )
            else:
                target_fn = plan_rrt_async
                thread_args = (
                    station,
                    q_current,
                    q_target,
                    vel_limits,
                    acc_limits,
                    traj_result,
                    args.step_size,
                    args.max_iter,
                )

            traj_thread = threading.Thread(
                target=target_fn,
                args=thread_args,
                daemon=True,
            )
            traj_thread.start()
            state = State.COMPUTING_PATH

        # ------------------------------------------------------------------
        elif state == State.COMPUTING_PATH:
            if not traj_result["ready"]:
                continue

            if not traj_result["success"]:
                next_idx = curr_solution_idx + 1
                print(
                    colored(
                        f"❌ RRT-Connect failed for solution {next_idx} "
                        f"(row {q_solutions[next_idx][0]}). Skipping.",
                        "yellow",
                    )
                )
                curr_solution_idx = next_idx
                state = State.IDLE
                continue

            # Always draw visuals once planning succeeds
            if traj_result["path"] is not None:
                plot_rrt_raw_path_in_meshcat(
                    station,
                    traj_result["path"],
                    name="rrt_raw_path",
                    rgba=Rgba(1.0, 0.4, 0.0, 1.0),  # orange
                )
            plot_trajectory_in_meshcat(
                station,
                traj_result["trajectory"],
                rgba=Rgba(0, 1, 1, 1),
                name="rrt_traj",
            )

            if preview_raw_mode:
                # Animate through raw RRT waypoints (forward + reverse) without committing
                raw_path = traj_result["path"]
                if raw_path is not None:
                    _animate_configs(
                        raw_path, station, station_context, simulator, meshcat
                    )
                print(
                    colored("✓ Raw path preview done. Ready for next action.", "cyan")
                )
                state = State.IDLE

            elif preview_smooth_mode:
                # Animate through TOPPRA trajectory samples (forward + reverse)
                spline = traj_result["trajectory"]
                if spline is not None:
                    ts = np.linspace(spline.start_time(), spline.end_time(), 50)
                    smooth_configs = [spline.value(t).flatten() for t in ts]
                    _animate_configs(
                        smooth_configs, station, station_context, simulator, meshcat
                    )
                print(
                    colored(
                        "✓ Smooth path preview done. Ready for next action.", "cyan"
                    )
                )
                state = State.IDLE

            else:
                # Execute — actually move the robot
                trajectory_start_time = simulator.get_context().get_time()
                print(colored("✓ RRT-Connect path found. Moving...", "green"))
                state = State.MOVING

        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        elif state == State.DONE:
            break

    print("Simulation ended.")


if __name__ == "__main__":
    main()
