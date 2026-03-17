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
from utils.iris import compute_iris_regions
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

    # ==================================================================
    # Output Directories
    # ==================================================================
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

    # Get total number of frames
    camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=camera_frame,
        length=0.1,
        radius=0.002,
        name="camera_link",
    )

    # Build diagram
    diagram = builder.Build()

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

    # region Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station, r, v)

    station_context = station.GetMyContextFromRoot(simulator.get_context())

    # Calculate IK for getting to the top of the hemisphere

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    prev_state = State.IDLE
    state = State.IDLE

    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    # Button management
    num_move_to_top_clicks = 0
    num_compute_iks_clicks = 0
    num_execute_traj_clicks = 0

    # IK computation thread state
    ik_thread = None
    # IK computation thread state
    ik_thread = None
    hemisphere_ik_result = {
        "ready": False,
        "valid_joints": True,
        "valid_velocities": True,
        "valid_collisions": True,
        "trajectory": None,
        "trajectory_start_time": None,
    }
    optical_axis_ik_result = {
        "ready": False,
        "valid_joints": True,
        "valid_velocities": True,
        "valid_collisions": True,
        "trajectory": None,
        "trajectory_start_time": None,
    }
    move_to_start_gcs_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
    }
    alternate_gcs_result = {
        "ready": False,
        "success": False,
        "trajectory": None,
    }

    # Initialize SEW Plane visualization (Actual)
    if show_sew_planes:
        station.internal_meshcat.SetObject(
            "sew_plane", Box(0.8, 0.8, 0.001), Rgba(0.0, 0.5, 1.0, 0.3)
        )
        # Initialize Reference Plane visualization (Psi = 0)
        station.internal_meshcat.SetObject(
            "ref_plane", Box(0.8, 0.8, 0.001), Rgba(1.0, 1.0, 1.0, 0.2)
        )

    # ---------------------------------------------------------------
    # Camera + two background threads
    #
    # _capture_thread  ←  owns camera.read() loop at camera's native rate.
    #                      Always stores the latest frame in _latest_frame.
    #                      Main loop NEVER calls camera.read().
    #
    # _writer_thread   ←  owns cv2.imwrite(); pulls from _frame_queue.
    #
    # At each pause the main loop just snapshots _latest_frame (a lock
    # grab, ~microseconds) and enqueues it → zero blocking on critical path.
    # ---------------------------------------------------------------
    camera = None
    _latest_frame = None
    _latest_frame_lock = None
    _capture_stop = None
    _frame_queue = None
    _capture_thread = None
    _writer_thread = None

    if not no_cam:
        camera = cv2.VideoCapture(4)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            print(
                colored(
                    "⚠ Could not open camera device 4 – frames will NOT be saved",
                    "yellow",
                )
            )

        _latest_frame_lock = threading.Lock()  # guards _latest_frame
        _capture_stop = threading.Event()  # set → capture thread exits
        _frame_queue: queue.Queue = queue.Queue(maxsize=0)  # unbounded

        def _capture_loop():
            """Runs in background; owns camera.read() entirely.
            Continuously updates _latest_frame so the main loop always has
            a fresh frame available without ever blocking on camera.read().
            Also keeps the camera driver buffer drained between pauses.
            """
            nonlocal _latest_frame
            while not _capture_stop.is_set():
                ret, frame = camera.read()  # blocks at camera rate (~33 ms/frame)
                if ret:
                    with _latest_frame_lock:
                        _latest_frame = frame  # overwrite; only the latest matters

        def _writer_loop():
            """Handles all cv2.imwrite() calls off the critical path."""
            while True:
                item = _frame_queue.get()
                if item is None:  # sentinel – exit
                    _frame_queue.task_done()
                    break
                frame_data, path_str = item
                cv2.imwrite(path_str, frame_data)
                _frame_queue.task_done()

        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
        _capture_thread.start()
        _writer_thread.start()
        print(colored("✓ Camera threads started", "cyan"))
    else:
        print(colored("✓ Camera disabled via --no_cam", "yellow"))

    # visualize all hemisphere waypoints
    T_cam_to_tip = np.array(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1],
        ]
    )
    T_cam_to_tip = RotationMatrix(T_cam_to_tip)
    T_cam_to_tip = RigidTransform(T_cam_to_tip)

    for i, wp in enumerate(hemisphere_waypoints):
        draw_triad(
            station.internal_meshcat,
            f"hemisphere_waypoint_{i}",
            wp @ T_cam_to_tip,
            length=0.02,
            radius=0.001,
            opacity=0.5,
        )

    # Per-scan metadata
    scan_frame_dir = None  # Path to current scan's frame folder
    optical_halfway_time = None  # traj_time at which inward pass ends

    # Per-scan stop-and-shoot state (initialised in COMPUTING_IKS)
    capture_traj_times: np.ndarray = np.array([])  # traj_times at each photo stop
    next_capture_idx: int = 0  # which stop we're waiting to reach
    scan_frame_idx: int = 0  # frame counter within the scan
    is_pausing_for_capture: bool = False
    pause_start_sim_time: float = 0.0
    hold_traj_time: float = 0.0  # trajectory time to hold at during pause

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state != prev_state:
            print(colored(f"State changed: {prev_state} -> {state}", "grey"))
            prev_state = state

        # Update SEW (Shoulder-Elbow-Wrist) Plane and dynamic PSI visualization
        internal_plant = station.get_internal_plant()
        internal_plant_context = station.get_internal_plant_context()

        # Joint 2, 4, 6 positions
        p_J2, p_J4, p_J6 = get_sew_joint_positions(
            internal_plant, internal_plant_context
        )

        # Use standalone functions from utils.sew_stereo
        R_WP_np, R_WR_np = compute_sew_and_ref_matrices(p_J2, p_J4, p_J6, r, v)
        psi_rad = compute_psi_from_matrices(R_WP_np, R_WR_np)

        if R_WP_np is not None and show_sew_planes:
            R_WP = RotationMatrix(R_WP_np)
            centroid = (p_J2 + p_J4 + p_J6) / 3.0
            station.internal_meshcat.SetTransform(
                "sew_plane", RigidTransform(R_WP, centroid)
            )

        if R_WR_np is not None:
            if show_sew_planes:
                R_WR = RotationMatrix(R_WR_np)
                station.internal_meshcat.SetTransform(
                    "ref_plane", RigidTransform(R_WR, (p_J2 + p_J6) / 2.0)
                )

            # Update PSI slider in real-time
            station.internal_meshcat.SetSliderValue(
                "Current PSI (deg)", np.rad2deg(psi_rad)
            )
        # ------------------------------------------------------------------

        if state == State.IDLE:
            if station.internal_meshcat.GetButtonClicks("Plan Move to Start") > 0:
                print(colored("Planning move to start (async)", "cyan"))

                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_initial = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )

                target_pose = hemisphere_waypoints[scan_idx - 1]
                target_rot = target_pose.rotation().matrix()
                target_pos = target_pose.translation()

                Q = kinematics_solver.IK_for_microscope(
                    target_rot, target_pos, psi=elbow_angle
                )

                # Step 2) Find IK closest to current joint values
                q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )
                q_des = kinematics_solver.find_closest_solution(Q, q_curr)

                # # Start background thread for GCS planning
                # move_to_start_gcs_result["ready"] = False
                # move_to_start_gcs_thread = threading.Thread(
                #     target=solve_gcs_traj_opt_async,
                #     args=(
                #         station,
                #         q_initial,
                #         q_des,
                #         vel_limits,
                #         acc_limits,
                #         move_to_start_gcs_result,
                #         1.0,  # duration_cost
                #         1.0,  # path_length_cost
                #         True, # visualize_solving
                #     ),
                #     daemon=True,
                # )
                # move_to_start_gcs_thread.start()
                # state = State.COMPUTING_MOVE_TO_START

                # Test IRIS
                regions = compute_iris_regions(station)

        elif state == State.COMPUTING_MOVE_TO_START:
            if move_to_start_gcs_result["ready"]:
                if move_to_start_gcs_result["success"]:
                    initial_trajectory = move_to_start_gcs_result["trajectory"]

                    plot_trajectory_in_meshcat(
                        station,
                        initial_trajectory,
                        rgba=Rgba(1, 0.5, 0, 1),
                        name="move_to_start_trajectory",
                    )

                    print(
                        colored(
                            "✓ GCS planning for start move complete. Moving now...",
                            "green",
                        )
                    )
                    trajectory_start_time = simulator.get_context().get_time()
                    state = State.PLANNING_MOVE_TO_START
                else:
                    print(colored("❌ GCS planning failed!", "red"))
                    state = State.IDLE

        elif state == State.PLANNING_MOVE_TO_START:
            if station.internal_meshcat.GetButtonClicks("Move to Start") > 0:
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_START
        elif state == State.WAITING_FOR_NEXT_SCAN:
            # if (
            #     station.internal_meshcat.GetButtonClicks("Compute IKs")
            #     > num_compute_iks_clicks
            # ):
            #     num_compute_iks_clicks = station.internal_meshcat.GetButtonClicks(
            #         "Compute IKs"
            #     )
            # else:
            #     continue

            if scan_idx >= len(hemisphere_waypoints):
                print(colored("✓ All scans complete!", "green"))
                state = State.DONE
                continue

            print(colored(f"Preparing trajectory for scan #{scan_idx}", "grey"))

            pose_curr = hemisphere_waypoints[scan_idx - 1]  # We assume scan_idx >= 1
            pose_target = hemisphere_waypoints[scan_idx]

            # # Visualize the target scan point as a triad for reference
            pose_target = pose_target @ T_cam_to_tip

            draw_triad(
                station.internal_meshcat,
                "next_scan_target",
                pose_target,
                length=0.1,
                radius=0.002,
            )

            # Get current end-effector pose from actual robot state
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            internal_plant = station.get_internal_plant()
            internal_plant_context = station.get_internal_plant_context()
            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            # Setup trajectory IK computations in background threads to avoid blocking the main simulation loop
            print(colored("Starting IK computation in background...", "yellow"))
            hemisphere_ik_result["ready"] = False
            hemisphere_ik_thread = threading.Thread(
                target=compute_hemisphere_traj_async,
                args=(
                    station,
                    hemisphere_pos,
                    hemisphere_radius,
                    hemisphere_axis,
                    eef_pose,
                    pose_target,
                    kinematics_solver,
                    q_curr,
                    elbow_angle,
                    hemisphere_ik_result,
                    True,  # plot_trajectories
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                    speed_factor,
                    max_joint_velocities,
                ),
                daemon=True,
            )
            hemisphere_ik_thread.start()

            optical_axis_ik_result["ready"] = False
            optical_axis_ik_thread = threading.Thread(
                target=compute_optical_axis_traj_async,
                args=(
                    station,
                    pose_curr,
                    kinematics_solver,
                    q_curr,
                    elbow_angle,
                    optical_axis_ik_result,
                    True,  # plot_trajectories
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                    distance_along_optical_axis,
                    speed_factor,
                    max_joint_velocities,
                ),
                daemon=True,
            )
            optical_axis_ik_thread.start()

            state = State.COMPUTING_IKS

        elif state == State.COMPUTING_IKS:
            # Wait for IK computation to complete
            if hemisphere_ik_result["ready"] and optical_axis_ik_result["ready"]:
                if not hemisphere_ik_result["valid_joints"]:
                    print(
                        colored(
                            "❌ Invalid joint values. Taking alternate path to next scan point, after optical axis trajectory is computed",
                            "yellow",
                        )
                    )
                    # state = State.MOVING_DOWN_OPTICAL_AXIS
                    # continue

                if not hemisphere_ik_result["valid_velocities"]:
                    print(
                        colored(
                            "❌ Invalid joint velocities. Taking alternate path to next scan point, after optical axis trajectory is computed",
                            "yellow",
                        )
                    )
                    # state = State.MOVING_DOWN_OPTICAL_AXIS
                    # continue

                if not optical_axis_ik_result["valid_joints"]:
                    print(
                        colored(
                            "❌ Invalid joint values in optical axis IK. Quitting program.",
                            "red",
                        )
                    )
                    break

                if not optical_axis_ik_result["valid_velocities"]:
                    print(
                        colored("❌ Invalid joint velocities. Quitting program.", "red")
                    )
                    break

                hemisphere_trajectory = hemisphere_ik_result["trajectory"]
                optical_axis_trajectory = optical_axis_ik_result["trajectory"]

                # Plotting already happened in the async functions (non-blocking)

                trajectory_start_time = simulator.get_context().get_time()
                print(
                    f"Simulator time when starting trajectory: {trajectory_start_time}"
                )
                print(colored("Starting trajectory execution", "yellow"))

                # ----------------------------------------------------------
                # Compute the halfway point of the optical axis trajectory
                # (the trajectory is mirrored: first half goes inward, second
                # half goes outward). Photos are taken only on the inward pass.
                # ----------------------------------------------------------
                optical_halfway_time = optical_axis_trajectory.end_time() / 2.0

                # Create per-scan output folder: outputs/scans/scan<NN>/
                scans_base = Path(__file__).parent.parent / "outputs" / "scans"
                scan_frame_dir = scans_base / f"scan{scan_idx:02d}"
                scan_frame_dir.mkdir(parents=True, exist_ok=True)
                midpoint_pose_saved = False
                print(colored(f"✓ Frame output dir: {scan_frame_dir}", "cyan"))

                # ----------------------------------------------------------
                # Build the stop-and-shoot schedule.
                # num_pictures evenly-spaced stops on [0, optical_halfway_time];
                # both endpoints (t=0 = start, t=halfway = deepest point) included.
                # ----------------------------------------------------------
                capture_traj_times = np.linspace(
                    0.0, optical_halfway_time, num_pictures
                )
                next_capture_idx = 0
                scan_frame_idx = 0
                is_pausing_for_capture = False
                pause_start_sim_time = 0.0
                hold_traj_time = 0.0
                print(
                    colored(
                        f"✓ Stop-and-shoot: {num_pictures} stops at "
                        f"{np.round(capture_traj_times, 2).tolist()}",
                        "cyan",
                    )
                )

                scan_idx += 1

                state = State.MOVING_DOWN_OPTICAL_AXIS
                # state = State.MOVING_ALONG_HEMISPHERE

        elif state == State.PLANNING_ALONG_ALTERNATE_PATH:
            print(colored("Planning alternate path (async)", "cyan"))

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            target_pose = hemisphere_waypoints[scan_idx]
            target_rot = target_pose.rotation().matrix()
            target_pos = target_pose.translation()

            Q = kinematics_solver.IK_for_microscope(
                target_rot, target_pos, psi=elbow_angle
            )

            q_des = kinematics_solver.find_closest_solution(Q, q_initial)

            # Start background thread for GCS planning
            alternate_gcs_result["ready"] = False
            alternate_gcs_thread = threading.Thread(
                target=solve_gcs_traj_opt_async,
                args=(
                    station,
                    q_current,
                    q_des,
                    vel_limits,
                    acc_limits,
                    alternate_gcs_result,
                    1.0,  # duration_cost
                    1.0,  # path_length_cost
                    True,  # visualize_solving
                ),
                daemon=True,
            )
            alternate_gcs_thread.start()
            state = State.COMPUTING_ALONG_ALTERNATE_PATH

        elif state == State.COMPUTING_ALONG_ALTERNATE_PATH:
            if alternate_gcs_result["ready"]:
                if alternate_gcs_result["success"]:
                    traj_to_next_scan = alternate_gcs_result["trajectory"]
                    print(
                        colored(
                            "✓ GCS planning for alternate path complete. Moving now...",
                            "green",
                        )
                    )
                    trajectory_start_time = simulator.get_context().get_time()
                    state = State.MOVING_ALONG_ALTERNATE_PATH
                else:
                    print(colored("❌ GCS planning failed!", "red"))
                    # Fallback or retry logic could go here
                    state = State.DONE  # Stop for now if planning fails

        elif state == State.MOVING_ALONG_ALTERNATE_PATH:
            # need to follow traj_to_start, and then traj_to_next_scan

            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= traj_to_next_scan.end_time():
                q_desired = traj_to_next_scan.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                scan_idx += 1  # Since we didn't increment at State.COMPUTING_IKS
                state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_TO_START:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= initial_trajectory.end_time():
                q_desired = initial_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                if scan_idx >= len(hemisphere_waypoints):
                    print(colored("✓ All scans complete!", "green"))
                    state = State.DONE
                else:
                    state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_ALONG_HEMISPHERE:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= hemisphere_trajectory.end_time():
                q_desired = hemisphere_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Hemisphere trajectory execution complete!", "green"))
                state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_DOWN_OPTICAL_AXIS:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if is_pausing_for_capture:
                # -------------------------------------------------------
                # PAUSED: hold robot at stop position while timer runs.
                # After 0.5 s: flush camera buffer, take one still, queue
                # it for async write, compensate trajectory_start_time.
                # -------------------------------------------------------
                q_desired = optical_axis_trajectory.value(hold_traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )

                elapsed_pause = current_time - pause_start_sim_time
                if elapsed_pause >= 0.5:
                    # Grab the latest frame the capture thread has buffered (if camera enabled).
                    # This is a lock snapshot (~microseconds) – no blocking read.
                    if not no_cam:
                        with _latest_frame_lock:
                            frame = (
                                _latest_frame.copy()
                                if _latest_frame is not None
                                else None
                            )
                        if frame is not None and scan_frame_dir is not None:
                            frame_path = str(
                                scan_frame_dir / f"frame_{scan_frame_idx:05d}.jpg"
                            )
                            _frame_queue.put((frame, frame_path))
                            scan_frame_idx += 1
                            print(
                                colored(
                                    f"  📷 Photo {scan_frame_idx}/{num_pictures} "
                                    f"at traj_t={hold_traj_time:.3f}s "
                                    f"→ frame_{scan_frame_idx-1:05d}.jpg",
                                    "cyan",
                                )
                            )

                    # Shift trajectory_start_time so the robot resumes
                    # from exactly the point it paused at.
                    trajectory_start_time += elapsed_pause
                    next_capture_idx += 1
                    is_pausing_for_capture = False

            elif traj_time <= optical_axis_trajectory.end_time():
                q_desired = optical_axis_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )

                # -----------------------------------------------------------
                # Check whether we've reached the next photo stop.
                # All stops are on the inward pass [0, optical_halfway_time].
                # -----------------------------------------------------------
                if (
                    next_capture_idx < len(capture_traj_times)
                    and traj_time >= capture_traj_times[next_capture_idx]
                ):
                    hold_traj_time = capture_traj_times[next_capture_idx]
                    pause_start_sim_time = current_time
                    is_pausing_for_capture = True
                    print(
                        colored(
                            f"  ⏸ Pausing at stop {next_capture_idx + 1}/{num_pictures} "
                            f"(traj_t={hold_traj_time:.3f}s)",
                            "yellow",
                        )
                    )

                # -----------------------------------------------------------
                # Save camera_link pose at the deepest inward point
                # (= optical_halfway_time, the last capture stop).
                # Saved once per scan on first crossing.
                # -----------------------------------------------------------
                if (
                    not midpoint_pose_saved
                    and scan_frame_dir is not None
                    and traj_time >= optical_halfway_time
                ):
                    internal_plant = station.get_internal_plant()
                    internal_plant_context = station.get_internal_plant_context()
                    camera_pose = internal_plant.GetFrameByName(
                        "camera_link"
                    ).CalcPoseInWorld(internal_plant_context)
                    pose_matrix = (
                        camera_pose.GetAsMatrix4()
                    )  # 4x4 homogeneous transform
                    pose_path = scan_frame_dir / "pose.npy"
                    np.save(str(pose_path), pose_matrix)
                    midpoint_pose_saved = True
                    print(
                        colored(
                            f"✓ Saved camera_link pose at deepest point "
                            f"t={traj_time:.3f}s → {pose_path}",
                            "cyan",
                        )
                    )

            elif traj_time <= optical_axis_trajectory.end_time() + 1:  # 1 s buffer
                pass
            else:
                print(colored("✓ Optical axis trajectory execution complete!", "green"))
                trajectory_start_time = simulator.get_context().get_time()

                # if (hemisphere_ik_result["valid_joints"] and hemisphere_ik_result["valid_velocities"]):
                #     state = State.MOVING_ALONG_HEMISPHERE
                # else:
                #     state = State.PLANNING_ALONG_ALTERNATE_PATH

                state = State.PLANNING_ALONG_ALTERNATE_PATH

        elif state == State.DONE:
            context = simulator.get_context()
            log = state_logger.FindLog(context)
            t = log.sample_times()
            data = log.data()  # shape: (num_states, num_samples)

            # stack time as first column
            out = np.vstack((t, data)).T

            joint_log_csv_path = (
                Path(__file__).parent.parent / "outputs" / "joint_log.csv"
            )
            np.savetxt(
                joint_log_csv_path,
                out,
                delimiter=",",
                header="time," + ",".join([f"x{i}" for i in range(data.shape[0])]),
                comments="",
            )

            # Also save as pickle
            log_data = {"sample_times": log.sample_times(), "data": log.data()}
            import pickle

            joint_log_pkl_path = (
                Path(__file__).parent.parent / "outputs" / "joint_log.pkl"
            )
            with open(joint_log_pkl_path, "wb") as f:
                pickle.dump(log_data, f)
            break
            # pass

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Start"
        )

        # num_compute_iks_clicks = station.internal_meshcat.GetButtonClicks(
        #     "Compute IKs"
        # )

        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        # Update joint sliders to show current joint positions
        station_context = station.GetMyContextFromRoot(simulator.get_context())
        q_current = station.GetOutputPort("iiwa.position_measured").Eval(
            station_context
        )
        for i in range(7):
            station.internal_meshcat.SetSliderValue(
                f"Joint {i+1} (deg)", np.rad2deg(q_current[i])
            )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Move to Start")
    station.internal_meshcat.DeleteButton("Move to Start")
    station.internal_meshcat.DeleteButton("Execute Trajectory")

    # Delete joint sliders
    for i in range(7):
        station.internal_meshcat.DeleteSlider(f"Joint {i+1} (deg)")

    # Shut down background threads cleanly.
    if not no_cam and camera is not None:
        # 1. Stop the capture thread (exits on next camera.read() return)
        _capture_stop.set()
        _capture_thread.join(timeout=5)
        camera.release()
        # 2. Drain and stop the writer thread
        _frame_queue.put(None)  # sentinel: tells writer thread to exit
        _frame_queue.join()  # wait for all queued imwrite()s to complete
        _writer_thread.join(timeout=10)
        print(colored("✓ Frame writer flushed and camera released", "grey"))


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
