"""
Check whether all IK solutions for a set of desired poses fall within at least
one IRIS region.

For each pose, IK_for_microscope() is called with a fixed elbow angle
(np.deg2rad(135)), yielding up to 8 SEW-based solutions.  Each solution is
tested against every loaded region via HPolyhedron.PointInSet().

A pose is considered "covered" if at least one of its IK solutions lies inside
at least one region.  For debugging, every solution that does NOT land in any
region is printed with its full joint-angle vector.

After the text report, an interactive Meshcat window opens with two sliders:
  - "Pose":        selects which pose to inspect (0-indexed)
  - "IK Solution": selects which IK solution for that pose to snap the robot to

Usage:
    python check_poses_in_iris.py --iris_yaml iris_regions_95.yaml
    python check_poses_in_iris.py --iris_yaml iris_regions_95.yaml --verbose
"""

import argparse
import sys
import time

from pathlib import Path

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    LoadIrisRegionsYamlFile,
    RigidTransform,
    RotationMatrix,
    Simulator,
)
from termcolor import colored

# ── repo-local imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.util.visualizations import draw_triad
from utils.kuka_geo_kin import KinematicsSolver


# ══════════════════════════════════════════════════════════════════════════════
# Define the poses you want to test here.
# Each entry is a RigidTransform (position + rotation).
# Edit this list to suit your debugging needs.
# ══════════════════════════════════════════════════════════════════════════════
def get_test_poses() -> list[RigidTransform]:
    """Return the list of desired end-effector (microscope-tip) poses to check."""
    poses = []

    # --- Example: hemisphere scan poses ---
    hemisphere_dist = 0.8
    hemisphere_angle = np.deg2rad(60)
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

    from utils.planning import generate_hemisphere_waypoints

    waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=20,
        coverage=0.40,
    )
    for wp in waypoints:
        poses.append(wp)

    return poses


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def in_any_region(q: np.ndarray, regions: list) -> bool:
    """Return True if q lies inside at least one HPolyhedron region."""
    return any(region.PointInSet(q) for region in regions)


def compute_all_ik(
    poses: list[RigidTransform], kin: KinematicsSolver, elbow_angle: float
) -> list[np.ndarray]:
    """Solve IK for every pose. Returns a list of Nx7 arrays (one per pose)."""
    print(colored(f"Computing IK for {len(poses)} poses ...", "cyan"))
    all_solutions = []
    for pose in poses:
        Q = kin.IK_for_microscope(
            pose.rotation().matrix(), pose.translation(), psi=elbow_angle
        )
        all_solutions.append(Q)
    return all_solutions


def check_poses(
    poses: list[RigidTransform],
    regions: list,
    all_ik_solutions: list[np.ndarray],
    elbow_angle: float,
    verbose: bool,
) -> list[int]:
    """
    For each pose:
      1. Use pre-computed IK solutions (up to 8).
      2. For each solution test membership in all regions.
      3. Report failures in detail.

    Returns list of failed pose indices.
    """
    n_poses = len(poses)
    n_regions = len(regions)
    print(
        colored(
            f"\nChecking {n_poses} poses against {n_regions} IRIS regions "
            f"(elbow angle = {np.rad2deg(elbow_angle):.1f} deg)\n",
            "cyan",
        )
    )

    total_solutions = 0
    total_failures = 0
    failed_poses = []

    for pose_idx, (pose, Q) in enumerate(zip(poses, all_ik_solutions)):
        p_0M = pose.translation()

        if Q.shape[0] == 0:
            print(
                colored(
                    f"  Pose {pose_idx:3d}: NO IK solutions found for "
                    f"p={np.round(p_0M, 4)}",
                    "yellow",
                )
            )
            failed_poses.append(pose_idx)
            continue

        pose_has_valid = False
        pose_failures = []

        for sol_idx in range(Q.shape[0]):
            q = Q[sol_idx]
            total_solutions += 1
            if in_any_region(q, regions):
                pose_has_valid = True
                if verbose:
                    print(
                        colored(
                            f"  Pose {pose_idx:3d}  sol {sol_idx}: "
                            f"IN region   q={np.round(q, 4)}",
                            "green",
                        )
                    )
            else:
                total_failures += 1
                pose_failures.append((sol_idx, q))
                if verbose:
                    print(
                        colored(
                            f"  Pose {pose_idx:3d}  sol {sol_idx}: "
                            f"NOT in any region  "
                            f"q={np.round(q, 4)}",
                            "red",
                        )
                    )

        # Always print failing solutions even if the pose is otherwise covered
        if not verbose and pose_failures:
            for sol_idx, q in pose_failures:
                print(
                    colored(
                        f"  Pose {pose_idx:3d}  sol {sol_idx}: "
                        f"NOT in any region  "
                        f"q={np.round(q, 4)}",
                        "red",
                    )
                )

        if not pose_has_valid:
            print(
                colored(
                    f"  Pose {pose_idx:3d}: ALL {Q.shape[0]} solutions "
                    f"outside regions  p={np.round(p_0M, 4)}",
                    "red",
                )
            )
            failed_poses.append(pose_idx)
        elif not verbose:
            n_pass = Q.shape[0] - len(pose_failures)
            n_fail = len(pose_failures)
            line = (
                f"  Pose {pose_idx:3d}: {Q.shape[0]} solutions — "
                f"{n_pass} in region, {n_fail} not in any region"
            )
            print(colored(line, "green" if n_fail == 0 else "yellow"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(colored("Summary:", "cyan"))
    print(f"  Poses checked       : {n_poses}")
    print(f"  IK solutions tested : {total_solutions}")
    print(f"  Solutions outside   : {total_failures}")
    print(f"  Poses with NO valid solution: {len(failed_poses)}")
    if failed_poses:
        print(colored(f"  Failed pose indices : {failed_poses}", "red"))
    else:
        print(
            colored(
                "  All poses have at least one valid IK solution in an IRIS region.",
                "green",
            )
        )
    print("=" * 70)

    return failed_poses


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Meshcat visualization
# ══════════════════════════════════════════════════════════════════════════════


def run_interactive_viz(
    station: IiwaHardwareStationDiagram,
    poses: list[RigidTransform],
    all_ik_solutions: list[np.ndarray],
    regions: list,
) -> None:
    """
    Open an interactive Meshcat window.  Two sliders let you pick a pose and an
    IK solution index; the robot snaps to that configuration instantly.

    Uses the optimization plant/diagram inside InternalStationDiagram because it
    has its own MeshcatVisualizer and a diagram context you can mutate freely.
    """
    opt_plant = station.get_optimization_plant()
    opt_diagram = station.internal_station.get_optimization_diagram()
    opt_ctx = station.internal_station.get_optimization_diagram_context()
    opt_plant_ctx = opt_diagram.GetMutableSubsystemContext(opt_plant, opt_ctx)
    iiwa_instance = opt_plant.GetModelInstanceByName("iiwa")
    meshcat = station.optimization_meshcat

    n_poses = len(poses)
    max_sols = max((Q.shape[0] for Q in all_ik_solutions if Q.shape[0] > 0), default=1)

    # Draw all hemisphere waypoints as triads
    for i, pose in enumerate(poses):
        draw_triad(
            meshcat,
            f"waypoints/wp_{i:03d}",
            pose,
            length=0.02,
            radius=0.001,
            opacity=0.5,
        )

    # Joint display sliders (read-only — updated in snap())
    joint_lower = opt_plant.GetPositionLowerLimits()
    joint_upper = opt_plant.GetPositionUpperLimits()
    for i in range(7):
        meshcat.AddSlider(
            f"Joint {i+1} (deg)",
            np.rad2deg(joint_lower[i]),
            np.rad2deg(joint_upper[i]),
            0.1,
            0,
        )

    meshcat.AddSlider("Pose", 0, n_poses - 1, 1, 0)
    meshcat.AddSlider("IK Solution", 0, max_sols - 1, 1, 0)
    meshcat.AddButton("Exit")

    print(colored(f"\nVisualization ready → {meshcat.web_url()}", "cyan"))
    print("Open that URL in a browser, use the sliders to inspect poses.")
    print("Press Ctrl-C or click Exit to quit.\n")

    prev_pose = -1
    prev_sol = -1

    def snap(pose_idx: int, sol_idx: int) -> None:
        Q = all_ik_solutions[pose_idx]
        p_0M = poses[pose_idx].translation()

        if Q.shape[0] == 0:
            print(colored(f"  Pose {pose_idx:3d}: no IK solutions", "yellow"))
            return

        # Clamp sol_idx to valid range for this pose
        sol_idx = min(sol_idx, Q.shape[0] - 1)
        q = Q[sol_idx]
        in_region = in_any_region(q, regions)
        status = "IN region" if in_region else "NOT in any region"
        color = "green" if in_region else "red"
        print(
            colored(
                f"  Pose {pose_idx:3d}  sol {sol_idx}  ({Q.shape[0]} total)  "
                f"p={np.round(p_0M, 3)}  {status}\n"
                f"    q={np.round(q, 4)}",
                color,
            )
        )

        opt_plant.SetPositions(opt_plant_ctx, iiwa_instance, q)
        opt_diagram.ForcedPublish(opt_ctx)

        for i in range(7):
            meshcat.SetSliderValue(f"Joint {i+1} (deg)", np.rad2deg(q[i]))

    # Initial draw
    snap(0, 0)

    try:
        while meshcat.GetButtonClicks("Exit") < 1:
            pose_idx = int(round(meshcat.GetSliderValue("Pose")))
            sol_idx = int(round(meshcat.GetSliderValue("IK Solution")))

            if pose_idx != prev_pose or sol_idx != prev_sol:
                prev_pose = pose_idx
                prev_sol = sol_idx
                snap(pose_idx, sol_idx)

            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    print(colored("Visualization closed.", "cyan"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def build_station():
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
            hemisphere_radius=0.100,
            use_hardware=False,
        ),
    )

    default_position = np.array([1.57079, 0.1, 0, -1.2, 0, 1.6, 0])
    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.position"))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.AdvanceTo(0.01)

    return station


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--iris_yaml",
        default="iris_regions_95.yaml",
        help="Path to the IRIS regions YAML file (default: iris_regions_95.yaml)",
    )
    parser.add_argument(
        "--elbow_deg",
        type=float,
        default=135.0,
        help="SEW elbow angle in degrees (default: 135)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every solution (passing and failing), not just failures",
    )
    args = parser.parse_args()

    iris_yaml = Path(args.iris_yaml)
    if not iris_yaml.is_absolute():
        iris_yaml = Path(__file__).parent / iris_yaml
    if not iris_yaml.exists():
        print(colored(f"ERROR: IRIS YAML not found: {iris_yaml}", "red"))
        sys.exit(1)

    elbow_angle = np.deg2rad(args.elbow_deg)

    # ── Load IRIS regions ────────────────────────────────────────────────────
    print(colored(f"Loading IRIS regions from: {iris_yaml}", "cyan"))
    iris_regions_dict = LoadIrisRegionsYamlFile(str(iris_yaml))
    regions = list(iris_regions_dict.values())
    print(colored(f"Loaded {len(regions)} regions.", "green"))

    # ── Build station ────────────────────────────────────────────────────────
    print(colored("Building station ...", "cyan"))
    station = build_station()

    r = np.array([0, 0, -1])
    v = np.array([0, 1, 0])
    kin = KinematicsSolver(station, r, v)
    print(colored("KinematicsSolver ready.", "green"))

    # ── Get poses to test ────────────────────────────────────────────────────
    poses = get_test_poses()
    print(colored(f"Testing {len(poses)} poses.", "cyan"))

    # ── Compute IK once for all poses ────────────────────────────────────────
    all_ik_solutions = compute_all_ik(poses, kin, elbow_angle)

    # ── Text-based check ─────────────────────────────────────────────────────
    check_poses(poses, regions, all_ik_solutions, elbow_angle, args.verbose)

    # ── Interactive Meshcat visualization ────────────────────────────────────
    run_interactive_viz(station, poses, all_ik_solutions, regions)


if __name__ == "__main__":
    main()
