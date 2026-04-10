import cv2
import argparse
import time
import os
from datetime import datetime


def play_video_stream(source=0, run_name=None):
    """
    Play a real-time video stream using OpenCV.
    
    Args:
        source: Video source (0 for default webcam, or path to video file)
        run_name: Optional name for the subfolder in outputs to save images
    """
    # Open the video capture
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    cap.set(cv2.CAP_PROP_FPS, 30)  # Attempt to set FPS to 30 if supported
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video stream opened: {width}x{height} @ {fps} FPS")
    print("Press 'q' to quit, 's' to save a snapshot, 'r' to toggle recording")
    
    snapshot_count = 0
    recording = False
    prev_time = time.time()
    
    # Create outputs directory if it doesn't exist
    outputs_root = os.path.join(os.getcwd(), 'outputs')
    if run_name:
        run_outputs_dir = os.path.join(outputs_root, run_name)
    else:
        # If no run_name, use a default folder (e.g., 'default_run')
        run_outputs_dir = os.path.join(outputs_root, 'default_run')
    os.makedirs(run_outputs_dir, exist_ok=True)

    # Subfolder management
    subfolder_index = 1
    def get_subfolder_path(idx):
        return os.path.join(run_outputs_dir, f"set_{idx:02d}")

    current_subfolder = get_subfolder_path(subfolder_index)
    os.makedirs(current_subfolder, exist_ok=True)

    # Button display properties (positioned under FPS)
    button_x, button_y = 10, 60
    button_width, button_height = 150, 40
    next_button_x, next_button_y = 10, 110
    next_button_width, next_button_height = 220, 40

    print("Press 'n' to move to the next subfolder for saving images.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame or stream ended")
            break
        
        # Create a clean copy for saving
        original_frame = frame.copy()

        # Calculate real-time FPS
        current_time = time.time()
        realtime_fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Add FPS text overlay to the frame
        fps_text = f"FPS: {realtime_fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if recording:
            # save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_filename = os.path.join(current_subfolder, f"frame_{timestamp}.jpg")
            cv2.imwrite(snapshot_filename, original_frame)

            # Draw recording indicator
            cv2.circle(frame, (width - 50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 110, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw screenshot button indicator (press 's' to use)
        cv2.rectangle(frame, (button_x, button_y), 
                  (button_x + button_width, button_y + button_height), 
                  (0, 255, 0), -1)
        cv2.putText(frame, "Press 'S'", (button_x + 20, button_y + 27), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw next subfolder button indicator (press 'n' to use)
        cv2.rectangle(frame, (next_button_x, next_button_y),
                  (next_button_x + next_button_width, next_button_y + next_button_height),
                  (255, 200, 0), -1)
        cv2.putText(frame, "Press 'N' for next set", (next_button_x + 10, next_button_y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Show current subfolder name
        cv2.putText(frame, f"Saving to: set_{subfolder_index:02d}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Video Stream', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('r'):
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'}")
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = os.path.join(current_subfolder, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_filename, original_frame)
            print(f"Screenshot saved: {snapshot_filename}")
            snapshot_count += 1
        elif key == ord('n'):
            subfolder_index += 1
            current_subfolder = get_subfolder_path(subfolder_index)
            os.makedirs(current_subfolder, exist_ok=True)
            print(f"Switched to subfolder: set_{subfolder_index:02d}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Play real-time video stream using OpenCV')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of the run (creates a subfolder in outputs)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit, otherwise keep as string (file path)
    source = int(args.source) if args.source.isdigit() else args.source
    
    play_video_stream(source, args.run_name)


if __name__ == "__main__":
    main()
