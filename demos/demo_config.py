import numpy as np

# Configuration for different modes
HARDWARE_CONFIG = {
    "speed_factor": 1.0,
    "max_joint_velocity_deg": 60.0,
}

SIMULATION_CONFIG = {
    "speed_factor": 5.0,
    "max_joint_velocity_deg": 1000.0,
}


def get_config(use_hardware: bool):
    cfg = HARDWARE_CONFIG if use_hardware else SIMULATION_CONFIG
    # Convert deg/s to rad/s for velocity limits
    cfg["max_joint_velocities"] = np.deg2rad(cfg["max_joint_velocity_deg"] * np.ones(7))
    return cfg
