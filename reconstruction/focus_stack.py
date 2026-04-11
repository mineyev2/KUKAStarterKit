import argparse
import glob
import os
import subprocess


def run_focus_stack_on_subfolders(parent_folder):
    # List all subdirectories in the parent folder
    # Create the output directory for focus-stacked results
    output_dir = os.path.join(parent_folder, "images")
    depth_dir = os.path.join(parent_folder, "depthmaps")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    excluded__search_dirs = {"images", "depthmaps", "colmap"}

    for subdir in sorted(os.listdir(parent_folder)):
        subdir_path = os.path.join(parent_folder, subdir)

        if os.path.isdir(subdir_path) and subdir not in excluded__search_dirs:
            # Find all jpg images in the subdirectory
            image_pattern = os.path.join(subdir_path, "*.jpg")
            image_files = sorted(glob.glob(image_pattern))
            if not image_files:
                print(f"No .jpg files found in {subdir_path}, skipping.")
                continue
            output_file = os.path.join(output_dir, f"{subdir}.jpg")
            depth_file = os.path.join(depth_dir, f"{subdir}.png")
            cmd = [
                "focus-stack",
                "--full-resolution-align",
                "--global-align",
                "--align-keep-size",
                "--no-contrast",
                f"--depthmap={depth_file}",
                f"--output={output_file}",
            ] + image_files
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running focus-stack in {subdir_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run focus-stack on all subfolders of a directory."
    )
    parser.add_argument(
        "parent_folder",
        type=str,
        help="Parent folder containing subdirectories of images.",
    )
    args = parser.parse_args()
    run_focus_stack_on_subfolders(args.parent_folder)


if __name__ == "__main__":
    main()
