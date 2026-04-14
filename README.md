# Installation

(TODO: Make sure to install CUDA and toolkits first)
(Make sure nvidia toolkit and CUDA are properly setup first)
This repo was run for:
CUDA 13.0
PyTorch for 13.0

```
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc
```

Make sure to install all submodules in repo first:
```
git submodule update --init --recursive
```

## Motion Planning Environment Setup

This repo uses Poetry for dependency management. To set up this project, first install
[Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10
installed on your system.

Then, configure poetry to set up a virtual environment that uses Python 3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the
following command:
```bash
poetry install -vvv
```
(the `-vvv` flag adds verbose output).

## Reconstruction Environment Installation
Before creating the reconstruction environment, ensure you have built colmap (cloned via submodule) with CUDA 13 and specify the correct architecture: [Colmap Installation](https://colmap.github.io/install.html)

```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qt6-base-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libmkl-full-dev
# Fix issue in Ubuntu's openimageio CMake config.
# We don't depend on any of openimageio's OpenCV functionality,
# but it still requires the OpenCV include directory to exist.
sudo mkdir -p /usr/include/opencv4
```
...

If you have ubuntu 22.04, do this first:
```
sudo apt-get install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
```

```
cd colmap
mkdir build
cd build
cmake .. -GNinja -DBLA_VENDOR=Intel10_64lp -DCMAKE_CUDA_ARCHITECTURES=120
ninja
sudo ninja install
```

No to "Use libmkl_rt.so as the default alternative to BLAS/LAPACK?" if asked

Creating the Python Environment:
```bash
cd reconstruction
uv sync
# Installing the colmap python bindings
cd ../colmap && uv pip install .
```

For local Drake and manipulation installations, insert the following at the end of the
`.venv/bin/activate` and `.venv/bin/activate.nu` files, modifying the paths and python
version as required:
```bash
export PYTHONPATH=""
export PYTHONPATH=~/drake-build/install/lib/python3.10/site-packages:${PYTHONPATH}
export PYTHONPATH=~/manipulation:${PYTHONPATH}
```

Activate the environment:
```bash
poetry shell
```

Install `git-lfs`:

```bash
git-lfs install
git-lfs pull
```

## Other Dependencies
You may also need to install the following:
```
sudo apt update
sudo apt install python3-tk
```

## iiwa Driver

Refer to our lab's `#hardware_kuka_iiwa` channel for setup details.

# Usage

## Object Scanning (`robot_scan/scan_object.py`)

Scans an object from hemisphere viewpoints. Pre-computes IK for all waypoints upfront, moves the robot along the hemisphere, and optionally descends along the optical axis at each waypoint to capture photos. If the hemisphere path for a waypoint is unsafe, RRT\*-Connect is used as a fallback automatically.

Images are saved to `microscope-data/scans/<YYYYMMDD_HHMMSS>/` along with a `scan_params.txt` describing all run parameters.

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--use_hardware` | no | off | Connect to the real iiwa robot instead of simulation |
| `--no_cam` | no | off | Disable camera capture (no images saved) |
| `--live_view` | no | off | Show live camera feed in a window while scanning |
| `--skip_opt` | no | off | Skip the optical axis trajectory — robot visits waypoints only, no photos |
| `--no_wait` | no | off | Execute every trajectory immediately without pressing "Execute Path" in Meshcat |
| `--start_idx` | no | `0` | Waypoint index to start from (useful for resuming a scan) |
| `--hemisphere_dist` | no | `0.8` | Distance from world origin to the hemisphere center in meters |
| `--hemisphere_angle` | no | `0.0` | Approach angle in degrees — rotates the hemisphere center and axis in the XY plane |
| `--hemisphere_radius` | no | `0.08` | Radius of the hemisphere scan surface in meters |
| `--hemisphere_z` | no | `0.36` | Z height of the hemisphere center in the world frame (meters) |

### Meshcat controls

Once running, open the Meshcat URL printed to the terminal. The following buttons are available:

| Button | Description |
|---|---|
| `Move to Scan` | Plan and execute the initial move to the first valid waypoint |
| `Execute Path` | Confirm and execute the planned hemisphere or RRT\* trajectory (skipped with `--no_wait`) |
| `Preview RRT* Raw` | Animate the raw RRT\* waypoints before committing (RRT\* fallback only) |
| `Preview RRT* Smooth` | Animate the TOPPRA-smoothed RRT\* trajectory before committing |
| `Stop Simulation` | Stop the scan loop cleanly |

### Outputs

| File | Description |
|---|---|
| `microscope-data/scans/<date>/scan_params.txt` | All run parameters |
| `microscope-data/scans/<date>/scan<NN>/frame_NNNNN.jpg` | Captured images per waypoint |
| `microscope-data/scans/<date>/scan<NN>/pose_NNNNN.npy` | Camera pose (4×4 matrix) at each captured frame |
| `outputs/joint_log.csv` | Full joint position log for the session |
| `outputs/hemisphere_q_solutions.csv` | Pre-computed joint configs for all waypoints |
| `outputs/hemisphere_q_failed_indices.npy` | Indices of waypoints where IK failed |
| `outputs/hemisphere_waypoints.png` | Plot of all generated hemisphere waypoints |

### Examples

**Simulation, step through each trajectory manually:**
```bash
python robot_scan/scan_object.py
```

**Hardware, run fully autonomously:**
```bash
python robot_scan/scan_object.py --use_hardware --no_wait
```

**Hardware, skip photos (motion only):**
```bash
python robot_scan/scan_object.py --use_hardware --skip_opt --no_wait
```

**Custom hemisphere geometry:**
```bash
python robot_scan/scan_object.py --hemisphere_radius 0.1 --hemisphere_z 0.4 --hemisphere_dist 0.7
```

**Resume from waypoint 10:**
```bash
python robot_scan/scan_object.py --start_idx 10 --no_wait
```

---

## Camera Calibration (`microscope_scripts/camera_calibration.py`)

Computes intrinsic camera parameters (matrix + distortion) from checkerboard images saved to disk and writes `camera_calib_intrinsic.npy` into the same folder as the input images.

Intended as the second step after capturing images with `calibrate_microscope.py`.

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_path` | yes | — | Path to folder containing checkerboard images (`.png` or `.jpg`) |
| `--size` | yes | — | Physical size of each checkerboard square in mm (e.g. `25`) |
| `--corners_h` | no | `8` | Number of internal corner intersections along the checkerboard height |
| `--corners_w` | no | `5` | Number of internal corner intersections along the checkerboard width |

> **Note on corners:** `--corners_h` and `--corners_w` are the number of *intersection points* (internal corners), not the number of squares. A board with 9×6 squares has **8×5** internal corners.

### Examples

**Calibrate from images captured by `calibrate_microscope.py`:**
```bash
cd microscope_scripts
python3 camera_calibration.py --data_path ../microscope-data/calibration/20260413_120000 --size 25
```

**Non-default checkerboard** (e.g. a board with 7×6 internal corners, 20 mm squares):
```bash
cd microscope_scripts
python3 camera_calibration.py --data_path /path/to/images --size 20 --corners_h 7 --corners_w 6
```

The script prints `[ok]` / `[miss]` for each image and reports the mean reprojection error in pixels when done. Lower is better — under 1.0 px is generally good.

---

## Reconstruction

To run an example to test that everything works, you can download their sample dataset. First, navigate to the `reconstruction` folder.

```
python datasets/download_dataset.py
```

Then, run `simple_trainer.py` with one of the datsets:
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden
```