import json
import os

from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap
import torch

from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _camera_model_name(cam) -> str:
    """Return the camera model as a plain string regardless of pycolmap version."""
    model = cam.model
    if hasattr(model, "name"):
        return model.name
    if hasattr(cam, "model_name"):
        return cam.model_name
    return str(model).split(".")[-1]


def _parse_camera(cam):
    """
    Return (K_3x3, distortion_params, camtype, model_name) for a pycolmap Camera.

    Parameter layout by model:
      SIMPLE_PINHOLE   [f, cx, cy]
      PINHOLE          [fx, fy, cx, cy]
      SIMPLE_RADIAL    [f, cx, cy, k1]
      RADIAL           [f, cx, cy, k1, k2]
      OPENCV           [fx, fy, cx, cy, k1, k2, p1, p2]
      OPENCV_FISHEYE   [fx, fy, cx, cy, k1, k2, k3, k4]
    """
    model_name = _camera_model_name(cam)
    p = np.array(cam.params, dtype=np.float64)

    if model_name == "SIMPLE_PINHOLE":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float64)
        distortion = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    elif model_name == "PINHOLE":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        distortion = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    elif model_name == "SIMPLE_RADIAL":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[3], 0.0, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "RADIAL":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[3], p[4], 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "OPENCV":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[4], p[5], p[6], p[7]], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "OPENCV_FISHEYE":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([p[4], p[5], p[6], p[7]], dtype=np.float32)
        camtype = "fisheye"
    else:
        raise ValueError(
            f"Unsupported camera model: {model_name}. "
            "Supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE"
        )

    return K, distortion, camtype, model_name


class Parser:
    """COLMAP parser compatible with pycolmap >= 4.0 (Reconstruction API)."""

    def __init__(self, data_dir, factor=1, normalize=False, test_every=8):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "colmap/sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "colmap/sparse")

        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        reconstruction = pycolmap.Reconstruction(colmap_dir)
        imdata = reconstruction.images

        valid_ids = [k for k, im in imdata.items() if im.has_pose]
        if len(valid_ids) == 0:
            raise ValueError("No images with a valid pose found in COLMAP.")

        w2c_mats, camera_ids = [], []
        Ks_dict, params_dict, imsize_dict, mask_dict, camtype_dict = {}, {}, {}, {}, {}
        bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

        for k in valid_ids:
            im = imdata[k]
            pose = im.cam_from_world()
            rot = pose.rotation.matrix()
            trans = pose.translation.reshape(3, 1)
            w2c_mats.append(
                np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
            )
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            if camera_id not in Ks_dict:
                cam = reconstruction.cameras[camera_id]
                K, distortion, camtype, model_name = _parse_camera(cam)
                K = K.copy()
                K[:2, :] /= factor
                Ks_dict[camera_id] = K
                params_dict[camera_id] = distortion
                camtype_dict[camera_id] = camtype
                imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
                mask_dict[camera_id] = None
                if model_name not in ("SIMPLE_PINHOLE", "PINHOLE"):
                    print(
                        f"Warning: camera {camera_id} uses {model_name}. Images have distortion."
                    )

        print(
            f"[Parser] {len(w2c_mats)} images, taken by {len(set(camera_ids))} cameras."
        )

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)
        image_names = [imdata[k].name for k in valid_ids]

        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        image_dir_suffix = (
            f"_{factor}"
            if (factor > 1 and not self.extconf["no_factor_suffix"])
            else ""
        )
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        point3D_ids = sorted(reconstruction.points3D.keys())
        point3D_id_to_idx = {pid: i for i, pid in enumerate(point3D_ids)}
        points = np.array(
            [reconstruction.points3D[pid].xyz for pid in point3D_ids], dtype=np.float32
        )
        points_err = np.array(
            [reconstruction.points3D[pid].error for pid in point3D_ids],
            dtype=np.float32,
        )
        points_rgb = np.array(
            [reconstruction.points3D[pid].color for pid in point3D_ids], dtype=np.uint8
        )

        image_id_to_name = {
            img_id: img.name for img_id, img in reconstruction.images.items()
        }
        point_indices: Dict[str, List[int]] = {}
        for pid in point3D_ids:
            p3d = reconstruction.points3D[pid]
            for elem in p3d.track.elements:
                img_name = image_id_to_name.get(elem.image_id)
                if img_name is not None:
                    point_indices.setdefault(img_name, []).append(
                        point3D_id_to_idx[pid]
                    )
        point_indices = {
            k: np.array(v, dtype=np.int32) for k, v in point_indices.items()
        }

        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            transform = T2 @ T1
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                T3 = np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    dtype=np.float64,
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_width = actual_width / colmap_width
        s_height = actual_height / colmap_height
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            w, h = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(w * s_width), int(h * s_height))

        self.mapx_dict, self.mapy_dict, self.roi_undist_dict = {}, {}, {}
        for camera_id, dist_params in self.params_dict.items():
            if len(dist_params) == 0:
                continue
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            camtype = camtype_dict[camera_id]
            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, dist_params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, dist_params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1, y1 = (grid_x - cx) / fx, (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + dist_params[0] * theta**2
                    + dist_params[1] * theta**4
                    + dist_params[2] * theta**6
                    + dist_params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)
                mask = (
                    (mapx > 0) & (mapy > 0) & (mapx < width - 1) & (mapy < height - 1)
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        self.scene_scale = np.max(
            np.linalg.norm(camera_locations - scene_center, axis=1)
        )


class Dataset:
    def __init__(
        self, parser: Parser, split="train", patch_size=None, load_depths=False
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        self.indices = (
            indices[indices % self.parser.test_every != 0]
            if split == "train"
            else indices[indices % self.parser.test_every == 0]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(
                image_name, np.empty(0, dtype=np.int32)
            )
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]
            depths = points_cam[:, 2]
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            data["points"] = torch.from_numpy(points[selector]).float()
            data["depths"] = torch.from_numpy(depths[selector]).float()

        return data
