"""
demos/joint_teleop.py

Interactive joint-space teleoperation via Meshcat sliders.

Each of the 7 iiwa joints has a slider in Meshcat. Moving a slider
immediately commands the robot to that joint position. Press 'Stop'
in Meshcat to exit.

Usage:
    python demos/joint_teleop.py              # simulation only
    python demos/joint_teleop.py --use_hardware  # real robot

WARNING: On hardware the robot tracks slider commands directly with no
trajectory smoothing or velocity limiting. Moving a slider quickly can
cause the robot to accelerate very fast and trigger a fault. Make small,
slow adjustments when running on the real iiwa.
"""

import argparse

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

    # default_position = np.deg2rad([88.65, 45.67, -26.69, -119.89, 9.39, -69.57, 15.66])

    default_position = np.deg2rad([-32.06, 56.57, 47.46, -115.28, -0.89, -70.31, -37.64])

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

    station.internal_meshcat.AddButton("Stop")
    print("Open Meshcat to control joints. Press 'Stop' to exit.")
    while station.internal_meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)


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
