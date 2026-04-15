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


def main() -> None:
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
            hemisphere_dist=0.8,
            hemisphere_angle=np.deg2rad(60),
            hemisphere_radius=0.08,
            use_hardware=False,
        ),
    )

    # default_position = np.deg2rad([88.65, 45.67, -26.69, -119.89, 9.39, -69.57, 15.66])

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

    station.internal_meshcat.AddButton("Stop")
    print("Open Meshcat to control joints. Press 'Stop' to exit.")
    while station.internal_meshcat.GetButtonClicks("Stop") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)


if __name__ == "__main__":
    main()
