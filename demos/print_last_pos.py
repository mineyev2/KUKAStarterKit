"""Print the last joint positions in a recorded .npz trajectory (in degrees)."""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True)
args = parser.parse_args()

data = np.load(args.npz)
pos_key = next((k for k in data if k in ("positions", "q", "pos")), None)
if pos_key is None:
    raise KeyError(f"No position key found. Available: {list(data.keys())}")

positions = data[pos_key]
if positions.ndim == 2 and positions.shape[1] == 7:
    q_last = positions[-1]
elif positions.ndim == 2 and positions.shape[0] == 7:
    q_last = positions[:, -1]
else:
    raise ValueError(f"Unexpected shape: {positions.shape}")

print("Last joint positions (deg):")
for i, v in enumerate(np.rad2deg(q_last)):
    print(f"  J{i+1}: {v:.3f}")
