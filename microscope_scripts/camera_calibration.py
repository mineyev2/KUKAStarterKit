#!/usr/bin/env python3
"""
Computes intrinsic camera parameters from checkerboard images saved to disk.

Reads all .jpg/.png images from a folder, detects checkerboard corners in each,
and runs cv2.calibrateCamera to produce the camera matrix and distortion
coefficients. Results are saved as camera_calib_intrinsic.npy into the
same folder as the input images.

Usage:
    python camera_calibration.py --data_path /path/to/images --size 25
    python camera_calibration.py --data_path /path/to/images --size 25 --corners_h 7 --corners_w 6
"""

import argparse
import glob
import os

import cv2 as cv
import numpy as np


def detect_corners(img, checkersize, corners_h=8, corners_w=5):
    """
    Detect checkerboard corners in a single image.

    Returns:
        found   (bool)       — whether corners were detected
        objp    (np.ndarray) — 3D world-frame corner coords (N,3), or None
        corners (np.ndarray) — refined 2D image-frame corner coords (N,1,2), or None
        img     (np.ndarray) — image with corners drawn if found
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, (corners_h, corners_w), None)

    if not found:
        return False, None, None, img

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((corners_h * corners_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_h, 0:corners_w].T.reshape(-1, 2) * checkersize

    cv.drawChessboardCorners(img, (corners_h, corners_w), corners, found)
    return True, objp, corners, img


def load_calibration_images(data_path, checkersize, corners_h=8, corners_w=5):
    """
    Load all images from data_path, run corner detection on each, and
    return the valid object/image point pairs.

    Returns:
        objpoints  (list)  — 3D world-frame points for each valid image
        imgpoints  (list)  — 2D image-frame points for each valid image
        image_size (tuple) — (width, height) of the images
    """
    images = sorted(
        glob.glob(os.path.join(data_path, "*.png"))
        + glob.glob(os.path.join(data_path, "*.jpg"))
    )

    if not images:
        raise FileNotFoundError(f"No .png or .jpg images found in: {data_path}")

    objpoints = []
    imgpoints = []
    image_size = None
    num_valid = 0

    for path in images:
        img = cv.imread(path)
        if img is None:
            print(f"  [skip] Could not read {path}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])  # (width, height)

        found, objp, corners, _ = detect_corners(img, checkersize, corners_h, corners_w)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            num_valid += 1
            print(f"  [ok]   {os.path.basename(path)}")
        else:
            print(f"  [miss] {os.path.basename(path)} — corners not found")

    print(f"\n{num_valid} / {len(images)} images had corners detected.")
    return objpoints, imgpoints, image_size


def main(
    data_path: str,
    checkersize: float,
    corners_h: int = 8,
    corners_w: int = 5,
) -> None:
    print(f"Loading images from: {data_path}")
    print(f"Square size: {checkersize} mm  |  Corners: {corners_h}h x {corners_w}w\n")

    objpoints, imgpoints, image_size = load_calibration_images(
        data_path, checkersize, corners_h, corners_w
    )

    if len(objpoints) < 3:
        raise RuntimeError(
            f"Need at least 3 valid images for calibration, only got {len(objpoints)}."
        )

    print("\nRunning calibration...")
    # flags = cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3
    flags = (
        cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST
    )
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    print(f"\nCamera matrix:\n{mtx}")
    print(f"\nDistortion coefficients:\n{dist}")

    # Reprojection error
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        mean_error += cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    print(f"\nMean reprojection error: {mean_error / len(objpoints):.4f} px")

    # Save next to the images
    calib_file = os.path.join(data_path, "camera_calib_intrinsic.npy")
    with open(calib_file, "wb") as f:
        np.save(f, mtx)
        np.save(f, dist)
    print(f"\nSaved intrinsic parameters to: {calib_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute intrinsic camera calibration from checkerboard images on disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_calibration.py --data_path calibration_data/calibration_images/25mm_calibration --size 25
  python camera_calibration.py --data_path /path/to/imgs --size 20 --corners_h 7 --corners_w 6
        """,
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to folder containing checkerboard images (.png or .jpg).",
    )
    parser.add_argument(
        "--size",
        type=float,
        required=True,
        help="Physical size of each checkerboard square in mm.",
    )
    parser.add_argument(
        "--corners_h",
        type=int,
        default=8,
        help="Number of internal corner intersections along the height (default: 8).",
    )
    parser.add_argument(
        "--corners_w",
        type=int,
        default=5,
        help="Number of internal corner intersections along the width (default: 5).",
    )

    args = parser.parse_args()
    main(
        data_path=args.data_path,
        checkersize=args.size,
        corners_h=args.corners_h,
        corners_w=args.corners_w,
    )
