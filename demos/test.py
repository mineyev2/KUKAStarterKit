from pathlib import Path

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Sphere,
)

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.visualizations import draw_triad
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import generate_hemisphere_waypoints
from utils.plotting import plot_hemisphere_waypoints
from utils.safety import check_collisions, check_joint_limits, filter_ik_solutions


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

    # ==================================================================
    # Parameters
    # ==================================================================
    hemisphere_dist = 0.8
    hemisphere_angle = np.deg2rad(60)
    hemisphere_pos = np.array(
        [
            hemisphere_dist * np.cos(hemisphere_angle),
            hemisphere_dist * np.sin(hemisphere_angle),
            0.36,
        ]
    )
    hemisphere_radius = 0.08
    hemisphere_axis = np.array(
        [-np.cos(hemisphere_angle), -np.sin(hemisphere_angle), 0]
    )

    num_scan_points = 100
    coverage = 1.0
    elbow_angle = np.deg2rad(135)
    default_position = np.deg2rad([88.65, 45.67, -26.69, -119.89, 9.39, -69.57, 15.66])

    r = np.array([0, 0, -1])
    v = np.array([0, 1, 0])

    T_cam_to_tip = RotationMatrix(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
    T_cam_to_tip = RigidTransform(T_cam_to_tip)

    # ==================================================================
    # Diagram Setup
    # ==================================================================
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

    kinematics_solver = KinematicsSolver(station, r, v)

    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.Initialize()

    plant = station.get_internal_plant()
    joint_lower_limits = plant.GetPositionLowerLimits()
    joint_upper_limits = plant.GetPositionUpperLimits()

    # ==================================================================
    # Waypoint Generation
    # ==================================================================
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=num_scan_points,
        coverage=coverage,
    )

    for i, wp in enumerate(hemisphere_waypoints):
        draw_triad(
            station.internal_meshcat,
            f"hemisphere_waypoint_{i}",
            wp @ T_cam_to_tip,
            length=0.02,
            radius=0.001,
            opacity=0.5,
        )

    heimsphere_waypoints_output_path = (
        Path(__file__).parent.parent / "outputs" / "hemisphere_waypoints.png"
    )
    plot_hemisphere_waypoints(
        hemisphere_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=heimsphere_waypoints_output_path,
        visualize=True,
    )

    # ==================================================================
    # Sequential IK Solving
    # ==================================================================
    n = len(hemisphere_waypoints)
    # Pre-allocate with NaN; failed rows stay NaN and are identifiable in the CSV
    q_array = np.full((n, 7), np.nan)
    q_prev = default_position.copy()
    failed_indices = []
    all_raw_solutions = []  # rows of [waypoint_idx, q1..q7] before filtering

    for i, wp in enumerate(hemisphere_waypoints):
        # pose = wp @ T_cam_to_tip
        pose = wp
        target_rot = pose.rotation().matrix()
        target_pos = pose.translation()

        Q = kinematics_solver.IK_for_microscope(target_rot, target_pos, psi=elbow_angle)

        # Record all raw solutions before filtering
        if Q is not None and Q.shape[0] > 0:
            for q_raw in Q:
                all_raw_solutions.append(np.concatenate([[i], q_raw]))

        # Filter out invalid solutions
        Q = filter_ik_solutions(
            station,
            Q,
            target_rot,
            target_pos,
            joint_lower_limits,
            joint_upper_limits,
        )
        # print("Q shape after filtering: " + str(Q.shape))

        if Q.shape[0] == 0:
            print(
                f"[{i}] FAIL: No valid IK solutions found for target_pos = {target_pos.round(3)}"
            )
            failed_indices.append(i)
            continue

        q_des = kinematics_solver.find_closest_solution(Q, q_prev)

        q_array[i] = q_des
        q_prev = q_des
        print(f"[{i}] q = {np.rad2deg(q_des).round(2)}")

    if failed_indices:
        print(f"\nFailed indices ({len(failed_indices)}): {failed_indices}")
    else:
        print("\nAll IK solutions found successfully.")

    # ==================================================================
    # Save as CSV
    # ==================================================================
    output_path = (
        Path(__file__).parent.parent / "outputs" / "hemisphere_q_solutions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, q_array, delimiter=",")
    n_success = int(np.sum(~np.isnan(q_array).any(axis=1)))
    print(
        f"Saved {n} rows ({n_success} valid, {n - n_success} failed/NaN) to {output_path}"
    )

    # Save all raw IK solutions (before filtering) — columns: [waypoint_idx, q1..q7]
    raw_output_path = Path(__file__).parent.parent / "outputs" / "hemisphere_q_raw.csv"
    if all_raw_solutions:
        raw_array = np.array(all_raw_solutions)
        np.savetxt(
            raw_output_path,
            raw_array,
            delimiter=",",
            header="waypoint_idx,q1,q2,q3,q4,q5,q6,q7",
            comments="",
            fmt=["%d"] + ["%.10f"] * 7,
        )
        print(f"Saved {len(all_raw_solutions)} raw IK solutions to {raw_output_path}")
    else:
        print("No raw IK solutions to save.")

    # Save waypoint results for offline viewing
    import json

    waypoints_data = {
        "hemisphere_pos": hemisphere_pos.tolist(),
        "hemisphere_radius": float(hemisphere_radius),
        "hemisphere_axis": hemisphere_axis.tolist(),
        "waypoints": [
            {"pos": wp.translation().tolist(), "rot": wp.rotation().matrix().tolist()}
            for wp in hemisphere_waypoints
        ],
        "failed_indices": failed_indices,
    }
    waypoints_json_path = (
        Path(__file__).parent.parent / "outputs" / "waypoints_result.json"
    )
    with open(waypoints_json_path, "w") as f:
        json.dump(waypoints_data, f, indent=2)
    print(f"Saved waypoint results to {waypoints_json_path}")

    while True:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)


if __name__ == "__main__":
    main()
