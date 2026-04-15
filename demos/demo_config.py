import numpy as np

# Configuration for different modes
HARDWARE_CONFIG = {
    "speed_factor": 1.0,
    "max_joint_velocity_deg": 30.0,
    "vel_limits": np.full(7, 0.5),  # rad/s
    "acc_limits": np.full(7, 0.5),  # rad/s^2
}

SIMULATION_CONFIG = {
    "speed_factor": 5.0,
    "max_joint_velocity_deg": 150.0,
    "vel_limits": np.full(7, 1.0),  # rad/s
    "acc_limits": np.full(7, 1.0),  # rad/s^2
}


def get_config(use_hardware: bool):
    cfg = HARDWARE_CONFIG if use_hardware else SIMULATION_CONFIG
    # Convert deg/s to rad/s for velocity limits
    cfg["max_joint_velocities"] = np.deg2rad(cfg["max_joint_velocity_deg"] * np.ones(7))
    return cfg
