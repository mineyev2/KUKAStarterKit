"""
demos/torque_teleop.py

Sanity-check script for torque control. Commands desired torques directly
to the iiwa joints instead of positions.

At zero slider offsets the robot holds its pose via gravity compensation
computed from the internal plant. Use the per-joint sliders to add torque
offsets and observe the arm respond.

How it works
------------
The scenario uses `control_mode: torque_only`, which exposes an
`iiwa.torque` input port (instead of `iiwa.position`). Each loop tick:

    tau_cmd = tau_slider_offset

Drake's simulated IiwaDriver replicates the real KUKA's internal gravity
compensation, so sending tau_cmd = 0 floats the arm in both simulation
and hardware. No explicit gravity compensation is needed here.

Usage:
    python demos/torque_teleop.py              # simulation
    python demos/torque_teleop.py --use_hardware  # real robot

Author: Roman Mineyev
"""

import argparse
import sys

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

# Initial joint configuration (radians)
DEFAULT_POSITION = np.deg2rad([-32.06, 56.57, 47.46, -115.28, -0.89, -70.31, -37.64])

# Per-joint torque slider range (Nm)
TORQUE_SLIDER_RANGE = 5.0


def main(use_hardware: bool = False) -> None:
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

    # -----------------------------------------------------------------------
    # Diagram setup
    # -----------------------------------------------------------------------
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

    # Dummy zero-torque source wired at build time; overridden via FixValue in the loop
    dummy = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.torque"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    # -----------------------------------------------------------------------
    # Simulator setup
    # -----------------------------------------------------------------------
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Set robot to default position in simulation by fixing the internal plant state
    # (only meaningful in sim; hardware starts wherever the arm currently is)
    if not use_hardware:
        station_ctx = station.GetMyMutableContextFromRoot(simulator.get_mutable_context())
        station.GetInputPort("iiwa.torque").FixValue(station_ctx, np.zeros(7))

    meshcat = station.internal_meshcat

    # -----------------------------------------------------------------------
    # Meshcat sliders — per-joint torque offsets
    # -----------------------------------------------------------------------
    for i in range(7):
        meshcat.AddSlider(
            f"J{i+1} torque offset (Nm)",
            -TORQUE_SLIDER_RANGE,
            TORQUE_SLIDER_RANGE,
            0.1,
            0.0,
        )

    meshcat.AddButton("Stop")

    # -----------------------------------------------------------------------
    # Terminal display
    # -----------------------------------------------------------------------
    LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
    COL_W  = 12
    DIVIDER = "─" * (6 + COL_W * 4 + 3 * 4)
    HEADER  = (f"{'':6s}"
               f"{'Pos (deg)':>{COL_W}}"
               f"{'Vel (°/s)':>{COL_W}}"
               f"{'τ cmd (Nm)':>{COL_W}}"
               f"{'τ ext (Nm)':>{COL_W}}")
    DISPLAY_LINES = 3 + len(LABELS)
    first_print = True

    def print_state(t, q, qd, tau_cmd, tau_ext):
        nonlocal first_print
        if not first_print:
            sys.stdout.write(f"\033[{DISPLAY_LINES}A")
        first_print = False
        lines = [DIVIDER, f"  t = {t:7.3f} s", HEADER]
        for i, label in enumerate(LABELS):
            lines.append(
                f"  {label}  "
                f"{np.rad2deg(q[i]):>{COL_W}.3f}"
                f"{np.rad2deg(qd[i]):>{COL_W}.3f}"
                f"{tau_cmd[i]:>{COL_W}.3f}"
                f"{tau_ext[i]:>{COL_W}.3f}"
            )
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    # Drake's simulated IiwaDriver replicates the real KUKA behavior: gravity
    # compensation is handled internally in both sim and hardware. Sending
    # tau_cmd = 0 floats the arm in both cases — no explicit compensation needed.

    print(colored("Torque teleop running. Sliders add per-joint torque offsets.", "cyan"))
    print(colored("At zero offsets the arm floats (KUKA handles gravity internally).", "cyan"))
    print()

    while meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01) # Runs at 100 Hz

        
        ctx = station.GetMyContextFromRoot(simulator.get_context())
        q       = station.GetOutputPort("iiwa.position_measured").Eval(ctx) # Current joint positions
        qd      = station.GetOutputPort("iiwa.velocity_estimated").Eval(ctx) # Current joint velocities
        tau_ext = station.GetOutputPort("iiwa.torque_external").Eval(ctx) # External torques (e.g. from gravity, contact)

        tau_cmd = np.array([ # Commanded torques sent to the robot, computed from slider offsets
            meshcat.GetSliderValue(f"J{i+1} torque offset (Nm)") for i in range(7)
        ])

        station_ctx = station.GetMyMutableContextFromRoot(simulator.get_mutable_context())
        station.GetInputPort("iiwa.torque").FixValue(station_ctx, tau_cmd)

        print_state(simulator.get_context().get_time(), q, qd, tau_cmd, tau_ext)

    meshcat.DeleteButton("Stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct torque control sanity check")
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        default=False,
        help="Connect to real iiwa hardware via LCM",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
