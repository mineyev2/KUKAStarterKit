# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A robotic 3D microscopy system using a KUKA LBR iiwa 14 R820 arm to autonomously scan objects from hemisphere viewpoints. The robot arm carries a microscope, captures images from multiple angles, and the results can be used for 3D reconstruction.

## Environment Setup

```bash
poetry env use python3.10
poetry install -vvv
poetry shell
```

If using a local Drake build, add to `.venv/bin/activate`:
```bash
export PYTHONPATH=~/drake-build/install/lib/python3.10/site-packages:${PYTHONPATH}
export PYTHONPATH=~/manipulation:${PYTHONPATH}
```

MOSEK license must be at `~/mosek/mosek.lic` for trajectory optimization.

## Running Demos

```bash
# Main scan demo (simulation)
python demos/scan_object_and_save_frames.py

# Main scan demo (real hardware — requires iiwa driver running)
python demos/scan_object_and_save_frames.py --use_hardware

# Replay pre-computed joint configs from CSV
python demos/replay_q_solutions.py --start_idx 0 --csv outputs/hemisphere_q_solutions.csv

# Interactive joint teleoperation
python demos/q_teleop.py

# IK test + waypoint generation
python demos/test.py
```

## Hardware Setup (Real Robot)

1. Start the Drake iiwa FRI driver: `bazel run //kuka-driver:kuka_driver`
2. (Optional) Start Schunk WSG gripper driver: `bazel run //src:schunk_driver`
3. Network: static IP `192.170.10.200` (iiwa), `192.168.1.200` (gripper)
4. Run demo with `--use_hardware` flag

## Architecture

### Key Modules

**`iiwa_setup/`** — Low-level robot interface (Drake systems, drivers, sensors):
- `iiwa/hardware_station.py`: `IiwaHardwareStationDiagram` — the central Drake diagram wrapping robot plant, scene graph, controllers, and Meshcat. Used in every demo.
- `util/traj_planning.py`: `solve_kinematic_traj_opt` / `solve_kinematic_traj_opt_async` — KinematicTrajectoryOptimization + TOPPRA reparameterization pipeline.
- `motion_planning/toppra.py`: `reparameterize_with_toppra` — wraps Drake's `Toppra` to produce a `PathParameterizedTrajectory` starting at `t=0`.
- `motion_planning/gcs.py`: GCS-based motion planning for collision-free alternate paths.

**`utils/`** — High-level planning and scanning logic:
- `planning.py`: Hemisphere waypoint generation (SLERP), trajectory execution (`move_along_trajectory`), GCS alternate path helpers, IRIS region computation, Meshcat visualization helpers.
- `safety.py`: Joint limit, velocity, and collision filtering of IK solutions.
- `kuka_geo_kin.py` + `geometric_subproblems.py` + `sew_stereo.py`: Geometry-based IK for the iiwa14 using the SEW (Spherical-Elbow-Wrist) parameterization — preferred over general-purpose IK for speed and reliability.
- `states.py`: `State` enum for all FSM states used in `scan_object_and_save_frames.py`.

**`demos/`** — Entry points. Each demo builds a Drake `DiagramBuilder`, wraps it in a `Simulator`, and runs a `while` loop advancing sim time in 0.05s steps while a state machine drives behavior.

**`models/`** — SDF/YAML model files for robot, microscope mount, and environment.

**`demo_config.py`** — Speed/velocity scaling: sim uses 5× speed and 1000 deg/s limits; hardware uses 1× and 60 deg/s.

### State Machine Pattern

All demos share the same structure:

```python
while not stop:
    simulator.AdvanceTo(t + 0.05)   # advance sim first, then check state
    if state == IDLE: ...
    elif state == COMPUTING_PATH:
        if traj_result["ready"]:
            trajectory_start_time = simulator.get_context().get_time()
            state = MOVING
    elif state == MOVING:
        done = move_along_trajectory(traj, trajectory_start_time, simulator, station)
```

Expensive planning (IK, GCS, trajectory optimization) always runs in background `threading.Thread`s, storing results in a shared dict with a `ready` flag. The main loop polls this flag while continuing to advance the simulator.

`trajectory_start_time` is always set to `simulator.get_context().get_time()` at the moment planning completes and state transitions to a MOVING state — never set before planning finishes.

### Trajectory Execution

`move_along_trajectory(traj, start_time, simulator, station)` in `utils/planning.py`:
- Computes `traj_time = current_sim_time - start_time`
- Calls `traj.value(traj_time)` and `FixValue` on the iiwa position port
- Returns `True` when `traj_time > traj.end_time()`

Trajectories from TOPPRA always start at `t=0`. `start_time` is a sim-clock offset.

### Output Artifacts

- `outputs/hemisphere_q_solutions.csv`: Joint configs per waypoint (NaN = IK failure)
- `outputs/scans/scan*/`: Captured microscope frames
- `outputs/joint_log.csv`: Full joint trajectory log
- `outputs/iris_regions.yaml`: Cached IRIS collision-free regions (reused across runs)
