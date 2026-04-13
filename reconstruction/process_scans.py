import argparse
import os
import shutil
import sqlite3

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from pycolmap import logging
from scipy.spatial.transform import Rotation as R
from visualizers.feature_viewer import FeatureViewer


def update_prior_poses(db_path, pose_dict, is_camera_to_world=False):
    """
    Updates the prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz
    fields in the COLMAP database.

    Args:
        db_path (str): Path to the COLMAP database.db
        pose_dict (dict): Dictionary mapping image names (as stored in the DB) to 4x4 numpy arrays.
        is_camera_to_world (bool): Set to True if your 4x4 matrices are Camera-to-World poses.
                                   COLMAP expects World-to-Camera.
    """
    # Connect to the COLMAP SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add prior pose columns if they don't exist
    columns = [
        "prior_qw",
        "prior_qx",
        "prior_qy",
        "prior_qz",
        "prior_tx",
        "prior_ty",
        "prior_tz",
    ]

    for col in columns:
        try:
            cursor.execute(f"ALTER TABLE images ADD COLUMN {col} REAL;")
        except sqlite3.OperationalError:
            # OperationalError is raised if the column already exists,
            # so we can safely ignore it.
            pass

    for image_name, matrix in pose_dict.items():
        # Invert the matrix if it's Camera-to-World
        if is_camera_to_world:
            matrix = np.linalg.inv(matrix)

        # Extract the 3x3 rotation matrix and 3x1 translation vector
        rot_matrix = matrix[:3, :3]
        tvec = matrix[:3, 3]

        # Convert rotation matrix to quaternion using Scipy
        # Note: Scipy's as_quat() returns scalar-last format: [qx, qy, qz, qw]
        rot = R.from_matrix(rot_matrix)
        qx, qy, qz, qw = rot.as_quat()

        tx, ty, tz = tvec

        # Update the row for this specific image in the database
        cursor.execute(
            """
            UPDATE images 
            SET prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?,
                prior_tx = ?, prior_ty = ?, prior_tz = ?
            WHERE name = ?
            """,
            (qw, qx, qy, qz, float(tx), float(ty), float(tz), image_name),
        )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    print(f"Successfully updated prior poses for {len(pose_dict)} images.")


def create_reference_reconstruction(db_path, output_dir):
    """
    Bypasses PyCOLMAP database bindings by reading directly via SQLite.
    Combines auto-assigned IDs with known poses to create a COLMAP text model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
    db_cameras = cursor.fetchall()

    cursor.execute(
        "SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images"
    )
    db_images = cursor.fetchall()
    conn.close()

    # Map COLMAP integer IDs to String Names
    MODEL_ID_TO_NAME = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE",
    }

    # Write cameras.txt
    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_id, model_id, width, height, params_blob in db_cameras:
            params = np.frombuffer(params_blob, dtype=np.float64)
            model_name = MODEL_ID_TO_NAME.get(model_id, "OPENCV")
            params_str = " ".join([str(p) for p in params])
            f.write(f"{cam_id} {model_name} {width} {height} {params_str}\n")

    # Write images.txt
    with open(output_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz in db_images:
            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n"
            )
            f.write("\n")

    # Write empty points3D.txt
    with open(output_dir / "points3D.txt", "w") as f:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Process microscope scans with COLMAP."
    )
    parser.add_argument(
        "--mode",
        choices=["triangulate", "automatic"],
        default="triangulate",
        help=(
            "'triangulate': use known robot poses to lock cameras and triangulate points. "
            "'automatic': run standard incremental SfM (poses estimated from images)."
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(
        "/home/codaero/Projects/Microscopic-3D-Reconstruction/microscope-data/scans/20260401_205638"
    )
    image_dir = dataset_path / "images"
    workspace_dir = dataset_path / "colmap" / args.mode
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True)

    database_path = workspace_dir / "database.db"

    print(f"Extracting features from {image_dir}...")
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.estimate_affine_shape = False  # high quality preset
    sift_options.max_num_features = 8192  # high quality preset

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift = sift_options

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "OPENCV"

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device="cuda",
    )

    feature_matching_options = pycolmap.FeatureMatchingOptions()
    feature_matching_options.use_gpu = True
    feature_matching_options.guided_matching = True

    pycolmap.match_exhaustive(
        database_path=database_path,
        matching_options=feature_matching_options,
        device="cuda",
    )

    if args.mode == "triangulate":
        # Use known robot poses: lock cameras, triangulate points.
        poses = {}
        for scan_dir in sorted(dataset_path.glob("scan*")):
            pose_file = scan_dir / "pose.npy"
            if pose_file.exists():
                image_name = f"{scan_dir.name}.jpg"
                poses[image_name] = np.load(pose_file)
                print(f"Loaded pose for {image_name}: shape {poses[image_name].shape}")

        update_prior_poses(database_path, poses, is_camera_to_world=True)

        reference_model_path = workspace_dir / "reference_model"
        create_reference_reconstruction(database_path, reference_model_path)

        reference = pycolmap.Reconstruction()
        reference.read(str(reference_model_path))

        triangulated_model_path = workspace_dir / "triangulated_model"
        triangulated_model_path.mkdir(exist_ok=True)

        print("Triangulating 3D points from known poses...")
        result = pycolmap.triangulate_points(
            reconstruction=reference,
            database_path=database_path,
            image_path=image_dir,
            output_path=triangulated_model_path,
            clear_points=True,
            refine_intrinsics=True,
        )
        print(result.summary())

    else:  # automatic
        # Standard automatic reconstruction SfM — poses estimated from images.
        sfm_path = workspace_dir / "sparse"
        if sfm_path.exists():
            shutil.rmtree(sfm_path)
        sfm_path.mkdir(exist_ok=True)

        mapping_options = pycolmap.IncrementalPipelineOptions()
        # ModifyForIndividualData
        mapping_options.min_focal_length_ratio = 0.1
        mapping_options.max_focal_length_ratio = 10.0
        mapping_options.max_extra_param = float("inf")
        # ModifyForHighQuality
        mapping_options.ba_local_max_num_iterations = 30
        mapping_options.ba_local_max_refinements = 3
        mapping_options.ba_global_max_num_iterations = 75
        mapping_options.ba_use_gpu = True

        print("Running automatic reconstruction (poses unknown)...")
        recs = pycolmap.incremental_mapping(
            database_path,
            image_dir,
            sfm_path,
            options=mapping_options,
        )
        for idx, rec in recs.items():
            logging.info(f"#{idx} {rec.summary()}")


if __name__ == "__main__":
    main()
