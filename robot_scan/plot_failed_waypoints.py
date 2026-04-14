"""Plot hemisphere waypoints twice: all waypoints and failed-only.

Loads hemisphere_q_failed_indices.npy from the outputs directory and
regenerates waypoints with the same parameters used during the scan.

Saves:
    outputs/hemisphere_waypoints_all.png    — all waypoints
    outputs/hemisphere_waypoints_failed.png — only the failed ones

Usage:
    python robot_scan/plot_failed_waypoints.py [options]

All geometry args must match what was used in scan_object.py.
"""

import argparse

from pathlib import Path

import numpy as np

from utils.planning import generate_hemisphere_waypoints
from utils.plotting import plot_hemisphere_waypoints


def main(
    hemisphere_dist: float = 0.8,
    hemisphere_angle_deg: float = 0.0,
    hemisphere_radius: float = 0.08,
    hemisphere_z: float = 0.36,
    num_scan_points: int = 50,
    coverage: float = 1.0,
) -> None:
    outputs_dir = Path(__file__).parent.parent / "outputs"

    failed_indices_path = outputs_dir / "hemisphere_q_failed_indices.npy"
    if not failed_indices_path.exists():
        raise FileNotFoundError(
            f"Could not find {failed_indices_path}. Run scan_object.py first."
        )
    failed_indices = np.load(failed_indices_path).astype(int).tolist()
    print(f"Loaded {len(failed_indices)} failed indices: {failed_indices}")

    hemisphere_angle = np.deg2rad(hemisphere_angle_deg)
    hemisphere_pos = np.array(
        [
            hemisphere_dist * np.cos(hemisphere_angle),
            hemisphere_dist * np.sin(hemisphere_angle),
            hemisphere_z,
        ]
    )
    hemisphere_axis = np.array(
        [-np.cos(hemisphere_angle), -np.sin(hemisphere_angle), 0]
    )

    all_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=num_scan_points,
        coverage=coverage,
    )

    # All waypoints
    plot_hemisphere_waypoints(
        all_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=outputs_dir / "hemisphere_waypoints_all.png",
    )

    # Failed waypoints only
    failed_waypoints = [all_waypoints[i] for i in failed_indices]
    if failed_waypoints:
        plot_hemisphere_waypoints(
            failed_waypoints,
            hemisphere_pos,
            hemisphere_radius,
            hemisphere_axis,
            output_path=outputs_dir / "hemisphere_waypoints_failed.png",
        )
    else:
        print("No failed waypoints — skipping hemisphere_waypoints_failed.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot all and failed hemisphere waypoints as separate PNGs."
    )
    parser.add_argument("--hemisphere_dist", type=float, default=0.8)
    parser.add_argument(
        "--hemisphere_angle", type=float, default=0.0, help="Approach angle in degrees."
    )
    parser.add_argument("--hemisphere_radius", type=float, default=0.08)
    parser.add_argument("--hemisphere_z", type=float, default=0.36)
    parser.add_argument("--num_scan_points", type=int, default=50)
    parser.add_argument("--coverage", type=float, default=1.0)
    args = parser.parse_args()

    main(
        hemisphere_dist=args.hemisphere_dist,
        hemisphere_angle_deg=args.hemisphere_angle,
        hemisphere_radius=args.hemisphere_radius,
        hemisphere_z=args.hemisphere_z,
        num_scan_points=args.num_scan_points,
        coverage=args.coverage,
    )
