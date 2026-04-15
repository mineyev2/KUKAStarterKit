"""
demos/joint_state_monitor.py

Joint teleop with live terminal state display.

Combines Meshcat joint sliders with a real-time terminal readout of all
7 joints showing position (deg), velocity (°/s), commanded torque (Nm),
and external torque (Nm). The display overwrites itself in place each
iteration so the terminal stays clean.

External torque is the residual between measured and model-predicted
torque — it spikes when the robot contacts something unexpected.

Usage:
    python demos/joint_state_monitor.py              # simulation
    python demos/joint_state_monitor.py --use_hardware  # real robot

WARNING: On hardware the robot tracks slider commands directly with no
trajectory smoothing or velocity limiting. Move sliders slowly.

Author: Roman Mineyev
"""

import argparse
import sys

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    Simulator,
)

from iiwa_setup.iiwa import IiwaHardwareStationDiagram


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
    
    default_position = np.array(
        [
            1.3648979733,
            1.7947597195,
            0.0000000000,
            0.0000000000,
            -1.9687825940,
            1.7235560643,
            1.7491270903,
        ]
    )

    controller_plant = station.get_iiwa_controller_plant()
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat, controller_plant, initial_value=default_position
        )
    )
    builder.Connect(teleop.get_output_port(), station.GetInputPort("iiwa.position"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # -----------------------------------------------------------------------
    # Terminal display helpers
    # -----------------------------------------------------------------------
    LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
    COL_W  = 12
    DIVIDER = "─" * (6 + COL_W * 4 + 3 * 4)
    HEADER  = (f"{'':6s}"
               f"{'Pos (deg)':>{COL_W}}"
               f"{'Vel (°/s)':>{COL_W}}"
               f"{'τ cmd (Nm)':>{COL_W}}"
               f"{'τ ext (Nm)':>{COL_W}}")
    DISPLAY_LINES = 3 + len(LABELS)  # divider + time + header + 7 joints
    first_print = True

    def print_state(t, q, qd, tau_cmd, tau_ext):
        nonlocal first_print
        if not first_print:
            sys.stdout.write(f"\033[{DISPLAY_LINES}A")  # move cursor up
        first_print = False

        lines = []
        lines.append(DIVIDER)
        lines.append(f"  t = {t:7.3f} s")
        lines.append(HEADER)
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

    station.internal_meshcat.AddButton("Stop")
    print("Open Meshcat to control joints. Press 'Stop' to exit.\n")
    while station.internal_meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

        ctx = station.GetMyContextFromRoot(simulator.get_context())
        q       = station.GetOutputPort("iiwa.position_measured").Eval(ctx)
        qd      = station.GetOutputPort("iiwa.velocity_estimated").Eval(ctx)
        tau_cmd = station.GetOutputPort("iiwa.torque_commanded").Eval(ctx)
        tau_ext = station.GetOutputPort("iiwa.torque_external").Eval(ctx)
        print_state(simulator.get_context().get_time(), q, qd, tau_cmd, tau_ext)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint-space teleoperation via Meshcat sliders")
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        default=False,
        help="Connect to real iiwa hardware via LCM",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
