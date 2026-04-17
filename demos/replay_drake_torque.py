"""
demos/replay_drake_torque.py

Load a recorded .npz trajectory and replay the tau_offset torque commands
in Drake simulation using torque_only control mode.

Expected npz keys:
    tau_offset : (N, 7) or (7, N) array of torque commands in Nm
    time_u     : (N,) array of timestamps in seconds

Buttons:
    "Start Trajectory" — begin replaying the loaded torques
    "Stop Simulation"  — exit

Usage:
    python demos/replay_drake_torque.py --npz path/to/traj.npz
    python demos/replay_drake_torque.py --npz path/to/traj.npz --use_hardware

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


def load_torque_trajectory(npz_path: str):
    data = np.load(npz_path)
    print(colored(f"NPZ keys: {list(data.keys())}", "cyan"))

    tau_key = next((k for k in data if k in ("tau_offset", "tau", "torque", "action")), None)
    time_key = next((k for k in data if k in ("time_u", "times", "t", "time", "timestamps")), None)

    if tau_key is None or time_key is None:
        raise KeyError(
            f"Could not find torque/time keys in npz. Available keys: {list(data.keys())}"
        )

    torques = data[tau_key]
    times = data[time_key].flatten()

    # Ensure shape is (7, N)
    if torques.ndim == 2 and torques.shape[1] == 7:
        torques = torques.T
    elif torques.ndim == 2 and torques.shape[0] == 7:
        pass
    else:
        raise ValueError(f"Unexpected torques shape: {torques.shape}. Expected (N,7) or (7,N).")

    if torques.shape[1] != len(times):
        raise ValueError(
            f"tau_offset has {torques.shape[1]} samples but times has {len(times)}."
        )

    print(colored(f"Loaded {len(times)} torque samples, duration={times[-1]-times[0]:.3f}s", "cyan"))

    times = times - times[0]

    traj = PiecewisePolynomial.FirstOrderHold(times, torques)
    tau0 = torques[:, 0]
    return traj, tau0


def main(use_hardware: bool, npz_path: str) -> None:
    traj, tau0 = load_torque_trajectory(npz_path)

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
            control_mode: torque_only
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
            control_mode="torque_only",
        ),
    )

    dummy = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.torque"))

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
                print(colored("Executing torque trajectory…", "cyan"))
                app_state = State.EXECUTING

        elif app_state == State.EXECUTING:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - traj_start_time

            if traj_time <= traj.end_time():
                tau_cmd = traj.value(traj_time).flatten()
                station_ctx = station.GetMyMutableContextFromRoot(simulator.get_mutable_context())
                station.GetInputPort("iiwa.torque").FixValue(station_ctx, tau_cmd)
            else:
                print(colored("✓ Torque trajectory complete.", "green"))
                app_state = State.DONE

        elif app_state == State.DONE:
            if meshcat.GetButtonClicks("Start Trajectory") > num_start_clicks:
                num_start_clicks += 1
                traj_start_time = simulator.get_context().get_time()
                print(colored("Replaying torque trajectory…", "cyan"))
                app_state = State.EXECUTING

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    print(colored("Simulation stopped.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded torque trajectory")
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to .npz file with 'tau_offset' (N,7) and 'time_u' (N,) arrays",
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        default=False,
        help="Connect to real iiwa hardware via LCM",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware, npz_path=args.npz)
