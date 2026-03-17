# General imports
import argparse
import queue
import threading

import matplotlib

matplotlib.use("Agg")

from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

# Personal files
from demo_config import get_config

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from mpl_toolkits.mplot3d import Axes3D
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    Box,
    ConstantVectorSource,
    DiagramBuilder,
    FrameIndex,
    GraphOfConvexSetsOptions,
    LogVectorOutput,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    SnoptSolver,
    Solve,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import (
    add_collision_constraints_to_trajectory,
    create_traj_from_q1_to_q2,
    resolve_gcs_with_toppra,
    resolve_with_toppra,
    setup_trajectory_optimization_from_q1_to_q2,
    setup_trajectory_optimization_from_q1_to_q2_without_collision_constraints,
)
from iiwa_setup.util.visualizations import draw_triad
from utils.iris import (
    compute_iris_regions,
    compute_iris_regions_v2,
    compute_iris_regions_v3,
)
from utils.kuka_geo_kin import KinematicsSolver
from utils.planning import (
    compute_hemisphere_traj_async,
    compute_optical_axis_traj_async,
    find_target_pose_on_hemisphere,
    generate_hemisphere_waypoints,
    generate_IK_solutions_for_path,
    generate_poses_along_hemisphere,
    generate_waypoints_down_optical_axis,
    hemisphere_slerp,
    plot_trajectory_in_meshcat,
    setup_gcs_traj_opt_from_q1_to_q2,
    solve_gcs_traj_opt,
    solve_gcs_traj_opt_async,
    sphere_frame,
)
from utils.plotting import (
    plot_hemisphere_waypoints,
    plot_path_with_frames,
    plot_trajectories_side_by_side,
    save_trajectory_plots,
)
from utils.sew_stereo import (
    compute_psi_from_matrices,
    compute_sew_and_ref_matrices,
    get_sew_joint_positions,
)


class State(Enum):
    IDLE = auto()
    WAITING_FOR_NEXT_SCAN = auto()
    PLANNING_MOVE_TO_START = auto()
    COMPUTING_MOVE_TO_START = auto()
    MOVING_TO_START = auto()
    MOVING_ALONG_HEMISPHERE = auto()
    MOVING_DOWN_OPTICAL_AXIS = auto()
    PLANNING_ALONG_ALTERNATE_PATH = auto()
    COMPUTING_ALONG_ALTERNATE_PATH = auto()
    MOVING_ALONG_ALTERNATE_PATH = auto()
    COMPUTING_IKS = auto()
    PAUSE = auto()
    DONE = auto()


def main(
    use_hardware: bool, no_cam: bool = False, show_sew_planes: bool = False
) -> None:
    # Load configuration
    cfg = get_config(use_hardware)
    speed_factor = cfg["speed_factor"]
    max_joint_velocities = cfg["max_joint_velocities"]

    # Clean up trajectory output folders
    import shutil

    hemisphere_paths_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    optical_axis_paths_dir = (
        Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    )

    if hemisphere_paths_dir.exists():
        shutil.rmtree(hemisphere_paths_dir)
        print(colored(f"✓ Cleared {hemisphere_paths_dir}", "grey"))

    if optical_axis_paths_dir.exists():
        shutil.rmtree(optical_axis_paths_dir)
        print(colored(f"✓ Cleared {optical_axis_paths_dir}", "grey"))

    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: worldhemisphere_radius
    #     child: sphere_obstacle::sphere_body
    #     X_PC:
    #         translation: [0.5, 0.0, 0.6]
    plant_config:
        # For some reason, this requires a small timestep
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
    # hemisphere_pos = np.array([0.0, 0.8, 0.36])
    hemisphere_pos = np.array(
        [
            hemisphere_dist * np.cos(hemisphere_angle),
            hemisphere_dist * np.sin(hemisphere_angle),
            0.36,
        ]
    )
    hemisphere_radius = 0.100
    hemisphere_axis = np.array(
        [-np.cos(hemisphere_angle), -np.sin(hemisphere_angle), 0]
    )

    num_scan_points = 50
    coverage = 0.40  # Fraction of hemisphere to cover
    distance_along_optical_axis = 0.025
    num_pictures = 1  # Default is 30
    elbow_angle = np.deg2rad(135)
    scan_idx = 1  # Default is 1

    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2

    r = np.array([0, 0, -1])
    v = np.array([0, 1, 0])

    T_tip_to_camera = np.eye(4)
    T_tip_to_camera[:3, 3] = [0, 0, 0.1]
    T_tip_to_camera[:3, :3] = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ]
    )

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_dist=hemisphere_dist,
            hemisphere_angle=hemisphere_angle,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    # Log joint positions using station's exported output port
    from pydrake.systems.primitives import VectorLogSink

    state_logger = builder.AddSystem(VectorLogSink(7))
    state_logger.set_name("state_logger")
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"), state_logger.get_input_port()
    )

    default_position = np.array([1.57079, 0.1, 0, -1.2, 0, 1.6, 0])
    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Build diagram
    diagram = builder.Build()

    # ==================================================================
    # Parameters
    # ==================================================================
    hemisphere_dist = 0.8
    hemisphere_angle = np.deg2rad(60)
    # hemisphere_pos = np.array([0.0, 0.8, 0.36])
    hemisphere_pos = np.array(
        [
            hemisphere_dist * np.cos(hemisphere_angle),
            hemisphere_dist * np.sin(hemisphere_angle),
            0.36,
        ]
    )
    hemisphere_radius = 0.100
    hemisphere_axis = np.array(
        [-np.cos(hemisphere_angle), -np.sin(hemisphere_angle), 0]
    )

    num_scan_points = 50
    coverage = 0.40  # Fraction of hemisphere to cover
    distance_along_optical_axis = 0.025
    num_pictures = 1  # Default is 30
    elbow_angle = np.deg2rad(135)
    scan_idx = 1  # Default is 1

    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2

    r = np.array([0, 0, -1])
    v = np.array([0, 1, 0])

    T_tip_to_camera = np.eye(4)
    T_tip_to_camera[:3, 3] = [0, 0, 0.1]
    T_tip_to_camera[:3, :3] = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ]
    )

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Plan Move to Start")
    station.internal_meshcat.AddButton("Move to Start")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Add joint position sliders (in degrees for readability)
    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    for i in range(7):
        min_deg = np.rad2deg(joint_lower_limits[i])
        max_deg = np.rad2deg(joint_upper_limits[i])
        station.internal_meshcat.AddSlider(
            f"Joint {i+1} (deg)", min_deg, max_deg, 0.1, 0
        )

    # Add PSI angle slider (read-only visualization)
    station.internal_meshcat.AddSlider("Current PSI (deg)", -180, 180, 0.1, 0)

    station_context = station.GetMyContextFromRoot(simulator.get_context())
    simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    regions = compute_iris_regions(station)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--no_cam",
        action="store_true",
        help="Disable camera threads (no frames will be saved).",
    )

    parser.add_argument(
        "--show_sew_planes",
        action="store_true",
        help="Show SEW and Reference planes in Meshcat.",
    )

    args = parser.parse_args()
    main(
        use_hardware=args.use_hardware,
        no_cam=args.no_cam,
        show_sew_planes=args.show_sew_planes,
    )
