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


def read_poses_from_scans(dataset_path, is_camera_to_world=False, as_quaternion=False):
    """
    Reads robot poses from each scan subdirectory.

    Args:
        dataset_path (Path): Root dataset directory containing scan* subdirectories.
        is_camera_to_world (bool): If True, the stored poses are Camera-to-World and
                                   will be inverted to World-to-Camera before returning.
        as_quaternion (bool): If True, each pose is returned as a dict
                              {'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'}.
                              If False, each pose is returned as a 4x4 numpy array.

    Returns:
        dict: Mapping image names (e.g. 'scan001.jpg') to either 4x4 numpy arrays
              or quaternion+translation dicts, depending on as_quaternion.
    """
    poses = {}
    for scan_dir in sorted(dataset_path.glob("scan*")):
        pose_file = scan_dir / "pose.npy"
        if pose_file.exists():
            image_name = f"{scan_dir.name}.jpg"
            matrix = np.load(pose_file)
            if is_camera_to_world:
                matrix = np.linalg.inv(matrix)
            if as_quaternion:
                qx, qy, qz, qw = R.from_matrix(matrix[:3, :3]).as_quat()
                tx, ty, tz = matrix[:3, 3]
                poses[image_name] = {
                    "qw": qw,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "tx": tx,
                    "ty": ty,
                    "tz": tz,
                }
            else:
                poses[image_name] = matrix
            fmt = "quaternion" if as_quaternion else "matrix"
            print(f"Loaded pose for {image_name} as {fmt}")
    return poses


def create_reference_reconstruction(db_path, output_dir, pose_dict):
    """
    Creates a COLMAP text model from the database and known poses.

    Args:
        db_path: Path to the COLMAP database.db
        output_dir: Directory to write cameras.txt, images.txt, points3D.txt
        pose_dict (dict): Mapping image names to quaternion+translation dicts
                          {'qw','qx','qy','qz','tx','ty','tz'}, as returned by
                          read_poses_from_scans(as_quaternion=True).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
    db_cameras = cursor.fetchall()

    cursor.execute("SELECT image_id, name, camera_id FROM images")
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
        for image_id, name, camera_id in db_images:
            pose = pose_dict[name]
            qw, qx, qy, qz = pose["qw"], pose["qx"], pose["qy"], pose["qz"]
            tx, ty, tz = pose["tx"], pose["ty"], pose["tz"]
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
        poses = read_poses_from_scans(
            dataset_path, is_camera_to_world=True, as_quaternion=True
        )

        reference_model_path = workspace_dir / "reference_model"
        create_reference_reconstruction(database_path, reference_model_path, poses)

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
