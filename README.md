This repo has a bunch of starter code to make it easy for you to use the KUKA iiwa 14 at the lab. SEW is used for IK in this code as well. Note that this is not fully cleaned yet so there's some things that are not fully implemented.

You can fork this repo for your personal projects as a starting place as well.

# Installation

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
poetry install
```

To run any code using this venv, you can do this:
```bash
poetry run python3 <PYTHON_FILE>
```

On the other hand, you can also just activate the venv:
```bash
eval $(poetry env activate)
```

## KUKA Hardware Setup

Refer to our lab's `#hardware_kuka_iiwa` channel for setup details.

# Usage

## Utility modules (`utils/`)

Don't rewrite what's already here — check these first:

| Module | What's in it |
|---|---|
| `kuka_geo_kin.py` | `KinematicsSolver` — geometric IK (`IK_for_microscope`, `IK_for_microscope_multiple_elbows`) and `find_closest_solution`. IK is parameterized via the SEW (Shoulder-Elbow-Wrist) stereo angle `ψ`, which controls elbow configuration. |
| `RRT.py` | `RRTConnect` bidirectional RRT planner, `plan_rrt_async`, `plot_rrt_raw_path_in_meshcat` |
| `RRTStar.py` | `RRTStarConnect` (extends RRT with rewiring), `plan_rrt_star_async` — drop-in replacement for RRT with better path quality |
| `planning.py` | Hemisphere waypoint generation, trajectory visualization (`plot_trajectory_in_meshcat`, `plot_configs_in_meshcat`), GCS trajectory optimization, `move_along_trajectory`, `compute_simple_traj_from_q1_to_q2` |
| `safety.py` | `check_joint_limits`, `check_joint_velocities`, `check_collisions`, `check_safety_constraints`, `filter_ik_solutions` — use before executing any trajectory |
| `sew_stereo.py` | SEW (Shoulder-Elbow-Wrist) arm angle parameterization: `compute_psi_from_matrices`, `get_sew_joint_positions`, `SEWStereo` class |
| `geometric_subproblems.py` | Low-level geometric subproblem solvers (SP0–SP4) used internally by the IK |
| `plotting.py` | Matplotlib helpers for hemisphere waypoints and trajectory plots (`plot_hemisphere_waypoints`, `save_trajectory_plots`) |
| `iris.py` | `compute_iris_regions` — collision-free IRIS region computation for GCS planning |
| `checkerboard_generator.py` | Generates a printable checkerboard PDF for camera calibration |
| `view_waypoints.py` | Standalone script to visualize saved waypoint JSON files in Meshcat |
| `states.py` | Shared `State` enum used across demos |

## Simulation environment

Every demo runs the iiwa inside a simulated world that approximates the
physical lab space. The world includes:

- **Floor** — a flat collision surface beneath the robot base.
- **2 walls** — placed to match the walls of the real lab bench area,
  preventing the planner from computing paths that would be blocked in
  the real environment.

These are registered as collision geometry, so RRT\* and trajectory
optimizers will automatically avoid them.

## Speed configuration (`demos/demo_config.py`)

All demos that plan trajectories read their speed limits from
`demos/demo_config.py`. Two profiles are defined:

| Parameter | Hardware | Simulation |
|---|---|---|
| `speed_factor` | 1.0 | 5.0 |
| `max_joint_velocity_deg` | 30 °/s | 150 °/s |
| `vel_limits` | 0.5 rad/s per joint | 1.0 rad/s per joint |
| `acc_limits` | 0.5 rad/s² per joint | 1.0 rad/s² per joint |

The correct profile is selected automatically based on the `--use_hardware`
flag. To slow the robot down — e.g. when testing a new path on hardware —
lower `vel_limits` and `acc_limits` in `HARDWARE_CONFIG`. These values are
passed directly to the RRT\* planner and TOPPRA reparameterizer, so reducing
them will produce longer but slower trajectories.

## Demo scripts
<!-- 
### `demos/eef_teleop.py` (hardware only)

End-effector pose teleoperation via Meshcat sliders. Six sliders (roll,
pitch, yaw, x, y, z) let you dial in a target pose; a trajectory is
planned automatically whenever the sliders change. Press **Execute
Trajectory** in Meshcat to run the path on the robot.

> The script only works for simulation right now.

```bash
python demos/eef_teleop.py
``` -->

### `demos/joint_teleop.py`

Joint-space teleoperation — one slider per joint. Commands are sent
directly with no smoothing, so move sliders slowly on hardware to avoid
faults.

```bash
python demos/joint_teleop.py              # simulation
python demos/joint_teleop.py --use_hardware  # real robot
```

### `demos/joint_state_monitor.py`

Joint teleop with a live terminal readout. Same Meshcat sliders as
`joint_teleop.py`, but also prints a real-time table of all 7 joints
that updates in place each tick:

```
──────────────────────────────────────────────────────────────────
  t = 422.700 s
         Pos (deg)   Vel (°/s)  τ cmd (Nm)  τ ext (Nm)
  J1       -48.114      -0.000       0.000       0.000
  J2       102.566      -0.000     -79.132       0.000
  J3         0.003      -0.000      -1.229       0.000
  J4        -0.002       0.000      23.271       0.000
  J5      -112.876       0.000      -1.095       0.000
  J6        98.555      -0.000       0.226       0.000
  J7       100.285      -0.000      -0.042       0.000
```

`τ ext` (external torque) is the residual between what the actuators
measure and what the dynamics model predicts — it spikes when the robot
contacts something unexpected, making this useful for basic contact
detection and hardware debugging.

```bash
python demos/joint_state_monitor.py
python demos/joint_state_monitor.py --use_hardware
```

### `demos/rrt_star_planner.py`

Plans an RRT\*-Connect trajectory between two hard-coded joint
configurations `Q1` and `Q2`. Press **Solve RRT\* Path** in Meshcat to
plan and visualize, then **Execute Path** to run it.

```bash
python demos/rrt_star_planner.py
python demos/rrt_star_planner.py --use_hardware
```

Feel free to add obstacles through `hardware_station.py` to test RRT*'s effectiveness.