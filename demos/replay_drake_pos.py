"""
demos/replay_drake_pos.py

Load a recorded .npz trajectory (joint positions + timestamps) and replay
it in Drake simulation.

Expected npz keys:
    positions : (N, 7) or (7, N) array of joint positions in radians
    times     : (N,) array of timestamps in seconds

Buttons:
    "Start Trajectory" — begin replaying the loaded trajectory
    "Stop Simulation"  — exit

Usage:
    python demos/replay_drake_pos.py --npz path/to/traj.npz
    python demos/replay_drake_pos.py --npz path/to/traj.npz --use_hardware

Author: Roman Mineyev
"""

import argparse
from enum import Enum, auto

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Simulator,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram


class State(Enum):
    IDLE = auto()
    EXECUTING = auto()
    DONE = auto()


def load_trajectory(npz_path: str):
    data = np.load(npz_path)
    print(colored(f"NPZ keys: {list(data.keys())}", "cyan"))

    # Flexible key lookup
    pos_key = next((k for k in data if k in ("positions", "q", "pos")), None)
    time_key = next((k for k in data if k in ("times", "t", "time", "timestamps", "time_q")), None)

    if pos_key is None or time_key is None:
        raise KeyError(
            f"Could not find position/time keys in npz. Available keys: {list(data.keys())}"
        )

    positions = data[pos_key]
    times = data[time_key].flatten()

    # Ensure shape is (7, N)
    if positions.ndim == 2 and positions.shape[1] == 7:
        positions = positions.T
    elif positions.ndim == 2 and positions.shape[0] == 7:
        pass
    else:
        raise ValueError(f"Unexpected positions shape: {positions.shape}. Expected (N,7) or (7,N).")

    if positions.shape[1] != len(times):
        raise ValueError(
            f"positions has {positions.shape[1]} samples but times has {len(times)}."
        )

    print(colored(f"Loaded {len(times)} waypoints, duration={times[-1]-times[0]:.3f}s", "cyan"))

    # Normalize times to start at 0
    times = times - times[0]

    traj = PiecewisePolynomial.CubicShapePreserving(times, positions)
    q0 = positions[:, 0]
    return traj, q0


def main(use_hardware: bool, npz_path: str) -> None:
    traj, q0 = load_trajectory(npz_path)

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

    builder = DiagramBuilder()
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            use_hardware=use_hardware,
        ),
    )

    dummy = builder.AddSystem(ConstantVectorSource(q0))
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
    meshcat.AddButton("Start Trajectory")
    meshcat.AddButton("Stop Simulation")

    app_state = State.IDLE
    prev_state = None
    traj_start_time = 0.0
    num_start_clicks = 0

    print(colored("Ready. Open Meshcat and press 'Start Trajectory' to begin.", "cyan"))
    print(colored(f"  Trajectory duration: {traj.end_time():.3f}s", "cyan"))

    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if app_state != prev_state:
            print(colored(f"  [{app_state.name}]", "grey"))
            prev_state = app_state

        if app_state == State.IDLE:
            if meshcat.GetButtonClicks("Start Trajectory") > num_start_clicks:
                num_start_clicks += 1
                traj_start_time = simulator.get_context().get_time()
                print(colored("Executing trajectory…", "cyan"))
                app_state = State.EXECUTING

        elif app_state == State.EXECUTING:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - traj_start_time

            if traj_time <= traj.end_time():
                q_desired = traj.value(traj_time)
                q_desired[6] = q0[6]  # lock joint 7
                station_ctx = station.GetMyMutableContextFromRoot(simulator.get_mutable_context())
                station.GetInputPort("iiwa.position").FixValue(station_ctx, q_desired)
            else:
                print(colored("✓ Trajectory complete.", "green"))
                app_state = State.DONE

        elif app_state == State.DONE:
            if meshcat.GetButtonClicks("Start Trajectory") > num_start_clicks:
                num_start_clicks += 1
                traj_start_time = simulator.get_context().get_time()
                print(colored("Replaying trajectory…", "cyan"))
                app_state = State.EXECUTING

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    print(colored("Simulation stopped.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded joint-position trajectory")
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to .npz file with 'positions' (N,7) and 'times' (N,) arrays",
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        default=False,
        help="Connect to real iiwa hardware via LCM",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware, npz_path=args.npz)
