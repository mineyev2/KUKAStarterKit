import argparse
import csv

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R


def plot_poses(poses_dict, output_html, axis_length=0.01):
    fig = go.Figure()

    origin = np.array([0, 0, 0, 1])
    x_tip = np.array([axis_length, 0, 0, 1])
    y_tip = np.array([0, axis_length, 0, 1])
    z_tip = np.array([0, 0, axis_length, 1])

    xs, ys, zs, names = [], [], [], []
    x_lines = [[], [], []]
    y_lines = [[], [], []]
    z_lines = [[], [], []]

    for name, p in poses_dict.items():
        o = (p @ origin)[:3]
        x = (p @ x_tip)[:3]
        y = (p @ y_tip)[:3]
        z = (p @ z_tip)[:3]

        xs.append(o[0])
        ys.append(o[1])
        zs.append(o[2])
        names.append(name)

        for lines, tip in [(x_lines, x), (y_lines, y), (z_lines, z)]:
            for i in range(3):
                lines[i].extend([o[i], tip[i], None])

    colors = {"x": "red", "y": "green", "z": "blue"}
    for (label, color), lines in zip(colors.items(), [x_lines, y_lines, z_lines]):
        fig.add_trace(
            go.Scatter3d(
                x=lines[0],
                y=lines[1],
                z=lines[2],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{label.upper()}-axis",
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker=dict(size=3, color="black"),
            text=names,
            textposition="top center",
            hoverinfo="text",
            name="Poses",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        title="Pose Viewer",
    )
    fig.write_html(output_html)
    print(f"Saved to {output_html}")


def export_pose_table(poses_dict, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz"])
        for name, p in poses_dict.items():
            tx, ty, tz = p[:3, 3]
            rot = R.from_matrix(p[:3, :3])
            qx, qy, qz, qw = rot.as_quat()  # scipy returns scalar-last
            writer.writerow([name, qw, qx, qy, qz, tx, ty, tz])
    print(f"Saved pose table to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing scan01/, scan02/, ...")
    parser.add_argument("--axis_length", type=float, default=0.005)
    args = parser.parse_args()

    dir_path = Path(args.directory)
    poses_dict = {}
    for scan_dir in sorted(dir_path.glob("scan*")):
        pose_file = scan_dir / "pose.npy"
        if pose_file.exists():
            poses_dict[scan_dir.name] = np.load(pose_file)

    if not poses_dict:
        print("No poses found.")
        return

    plot_poses(poses_dict, dir_path / "poses.html", axis_length=args.axis_length)
    export_pose_table(poses_dict, dir_path / "poses.csv")


if __name__ == "__main__":
    main()
