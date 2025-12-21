import argparse
import os
import platform
import subprocess
from typing import Tuple

import numpy as np


def parse_vector(vector_str: str) -> np.ndarray:
    cleaned = vector_str.replace("[", " ").replace("]", " ")
    vec = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    if vec.size == 0:
        raise ValueError("Failed to parse vector. Provide 84 numbers.")
    return vec


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    if "state_array" not in data or "action_array" not in data:
        raise KeyError("Expected state_array and action_array in npz.")
    return data["state_array"], data["action_array"]


def find_best_match(states: np.ndarray, target: np.ndarray) -> Tuple[int, float]:
    if states.shape[1] != target.shape[0]:
        raise ValueError(f"Target length {target.shape[0]} does not match states dim {states.shape[1]}.")
    diffs = states - target
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the closest frame for a given state vector.")
    parser.add_argument(
        "--vector",
        required=True,
        help="State vector as a space-separated string (84 values).",
    )
    parser.add_argument(
        "--data",
        default="data/imitation_state_and_action_data.npz",
        help="Path to npz file with state_array and action_array.",
    )
    parser.add_argument(
        "--frames",
        default="data/expert_frames",
        help="Path to frames folder.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the matched frame image.",
    )
    args = parser.parse_args()

    target = parse_vector(args.vector)
    states, actions = load_data(args.data)
    idx, dist = find_best_match(states, target)

    frame_name = f"frame_{idx}.jpg"
    frame_path = os.path.abspath(os.path.join(args.frames, frame_name))
    action = int(actions[idx])

    print(f"Closest match index: {idx}")
    print(f"Distance: {dist:.6f}")
    print(f"Action: {action}")
    print(f"Frame path: {frame_path}")
    if not os.path.exists(frame_path):
        print("Warning: frame file not found at that path.")
    elif args.open:
        system = platform.system().lower()
        if system == "darwin":
            subprocess.run(["open", frame_path], check=False)
        elif system == "linux":
            subprocess.run(["xdg-open", frame_path], check=False)
        else:
            print("Auto-open not supported on this OS.")


if __name__ == "__main__":
    main()
