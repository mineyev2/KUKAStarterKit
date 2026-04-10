"""
ik_from_q_z_offset.py

Given a user-defined joint configuration q, computes FK for the microscope
tip link, applies a z-offset along that frame's local z-axis, finds the
closest IK solution, then displays both poses in Meshcat and holds until
"Stop Simulation" is clicked.

Usage:
    python ik_from_q_z_offset.py
"""

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    MeshcatVisualizer,
    RigidTransform,
    Simulator,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from utils.kuka_geo_kin import KinematicsSolver

# ===========================================================================
# USER-DEFINED PARAMETERS
# ===========================================================================

# Starting joint configuration (radians)
# -30.18   59.88   47.46 -105.11   -1.3   -63.52  -39.02
# q = np.deg2rad([-30.18, 59.88, 47.46, -105.11, -1.3, -63.52, -39.02])
# -0.59040767  1.13747514  0.83692803 -1.80499397 -0.28445306 -1.12510898 -0.61232144
q = np.array(
    [
        -0.59040767,
        1.13747514,
        0.83692803,
        -1.80499397,
        -0.28445306,
        -1.12510898,
        -0.61232144,
    ]
)

# Z-offset in the microscope tip link's local frame (meters)
# Positive = along tip-frame +z, negative = along tip-frame -z
z_offset_m = 0.1

# Elbow angle for IK (radians)
elbow_angle = np.deg2rad(135)

# SEW parameterization vectors (match whatever you use elsewhere)
r = np.array([0, 0, -1])
v = np.array([0, 1, 0])

# ===========================================================================

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

# ---------------------------------------------------------------------------
# Build diagram
# ---------------------------------------------------------------------------
builder = DiagramBuilder()
scenario = LoadScenario(data=scenario_data)
station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
    "station",
    IiwaHardwareStationDiagram(
        scenario=scenario,
        use_hardware=False,
        hemisphere_dist=0.8,
        hemisphere_angle=np.deg2rad(0),
        hemisphere_radius=0.1,
    ),
)

_ = MeshcatVisualizer.AddToBuilder(
    builder, station.GetOutputPort("query_object"), station.internal_meshcat
)

diagram = builder.Build()

simulator = Simulator(diagram)
ApplySimulatorConfig(scenario.simulator_config, simulator)
simulator.set_target_realtime_rate(1.0)

station.internal_meshcat.AddButton("Stop Simulation")


def set_q(q_val):
    mutable_context = station.GetMyMutableContextFromRoot(
        simulator.get_mutable_context()
    )
    station.GetInputPort("iiwa.position").FixValue(mutable_context, q_val)


# ---------------------------------------------------------------------------
# Show robot at q
# ---------------------------------------------------------------------------
set_q(q)
simulator.AdvanceTo(0.1)

# ---------------------------------------------------------------------------
# FK: compute the microscope tip pose at q
# ---------------------------------------------------------------------------
plant = station.get_internal_plant()
plant_context = plant.CreateDefaultContext()
plant.SetPositions(plant_context, q)

tip_frame = plant.GetFrameByName("microscope_tip_link")
tip_pose: RigidTransform = plant.CalcRelativeTransform(
    plant_context, plant.world_frame(), tip_frame
)

print("=== FK at q ===")
print(f"  q (deg):             {np.round(np.rad2deg(q), 2)}")
print(f"  tip position (m):    {np.round(tip_pose.translation(), 4)}")
print(f"  tip rotation:\n{np.round(tip_pose.rotation().matrix(), 4)}")

# ---------------------------------------------------------------------------
# Apply z-offset in the tip frame's local z-axis
# ---------------------------------------------------------------------------
target_pose: RigidTransform = tip_pose @ RigidTransform(
    np.array([0.0, 0.0, z_offset_m])
)

print(f"\n=== Target pose (z_offset = {z_offset_m * 1000:.1f} mm) ===")
print(f"  target position (m): {np.round(target_pose.translation(), 4)}")

# ---------------------------------------------------------------------------
# IK: find closest joint config for the target pose
# ---------------------------------------------------------------------------
kinematics_solver = KinematicsSolver(station, r, v)

Q = kinematics_solver.IK_for_microscope(
    target_pose.rotation().matrix(), target_pose.translation(), psi=elbow_angle
)

if Q is None or len(Q) == 0:
    print(colored("\n[FAIL] No IK solution found — displaying q only.", "red"))
else:
    q_sol = kinematics_solver.find_closest_solution(Q, q)
    print(colored("\n=== IK solution (closest to q) ===", "green"))
    print(f"  q_sol (rad):   {q_sol})")

    set_q(q_sol)
    simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

# ---------------------------------------------------------------------------
# Hold until "Stop Simulation" is clicked
# ---------------------------------------------------------------------------
print(colored('\nHolding — click "Stop Simulation" in Meshcat to exit.', "cyan"))
while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
    simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)

station.internal_meshcat.DeleteButton("Stop Simulation")
