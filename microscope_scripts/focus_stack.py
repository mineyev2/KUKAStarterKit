import os
import subprocess
import argparse
import glob

def run_focus_stack_on_subfolders(parent_folder):
    # List all subdirectories in the parent folder
    # Create the output directory for focus-stacked results
    output_dir = os.path.join(parent_folder, "focus_stacked")
    os.makedirs(output_dir, exist_ok=True)

    for subdir in sorted(os.listdir(parent_folder)):
        subdir_path = os.path.join(parent_folder, subdir)
        if os.path.isdir(subdir_path) and subdir != "focus_stacked":
            # Find all jpg images in the subdirectory
            image_pattern = os.path.join(subdir_path, "*.jpg")
            image_files = sorted(glob.glob(image_pattern))
            if not image_files:
                print(f"No .jpg files found in {subdir_path}, skipping.")
                continue
            output_file = os.path.join(output_dir, f"{subdir}.jpg")
            cmd = [
                "focus-stack",
                "--align-keep-size",
                "--global-align",
                f"--output={output_file}"
            ] + image_files
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running focus-stack in {subdir_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run focus-stack on all subfolders of a directory.")
    parser.add_argument("parent_folder", type=str, help="Parent folder containing subdirectories of images.")
    args = parser.parse_args()
    run_focus_stack_on_subfolders(args.parent_folder)

if __name__ == "__main__":
    main()
