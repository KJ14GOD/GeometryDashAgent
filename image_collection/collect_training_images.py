"""
Image Collection Script for Geometry Dash YOLO Training

Captures frames from Geometry Dash every 2 seconds and saves them
to a folder for labeling and training.
"""
from mss import mss
import cv2 as cv
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path


def collect_images(output_folder="geometry_dash_dataset", interval_seconds=2, max_images=200):
    """
    Collect training images from Geometry Dash.
    
    Args:
        output_folder: Folder name to save images (on Desktop)
        interval_seconds: Time between captures (default: 2 seconds)
        max_images: Maximum number of images to collect
    """
    # Setup monitor region (same as screen_capture.py)
    monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
    
    # Create output folder on Desktop
    desktop_path = Path.home() / "Desktop"
    dataset_folder = desktop_path / output_folder / "images"
    dataset_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Geometry Dash Image Collection")
    print("=" * 60)
    print(f"Output folder: {dataset_folder}")
    print(f"Capture interval: {interval_seconds} seconds")
    print(f"Max images: {max_images}")
    print(f"Monitor region: {monitor['width']}x{monitor['height']}")
    print("\nStarting in 3 seconds...")
    print("Make sure Geometry Dash is running!")
    print("Press Ctrl+C to stop early")
    print("=" * 60)
    time.sleep(3)
    
    # Initialize screen capture
    sct = mss()
    
    # Frame counter
    frame_count = 0
    start_time = time.time()
    last_capture_time = time.time()
    
    try:
        while frame_count < max_images:
            current_time = time.time()
            
            # Check if it's time to capture (every interval_seconds)
            if current_time - last_capture_time >= interval_seconds:
                # Capture frame
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                
                # Convert BGRA to BGR
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
                
                # Generate filename with timestamp and counter
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stereo_madness_{frame_count+1:04d}_{timestamp}.jpg"
                filepath = dataset_folder / filename
                
                # Save image
                cv.imwrite(str(filepath), img)
                
                frame_count += 1
                elapsed = current_time - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"[{frame_count}/{max_images}] Saved: {filename} "
                      f"(Elapsed: {elapsed:.1f}s, Avg: {avg_fps:.2f} img/s)")
                
                last_capture_time = current_time
            
            # Small sleep to avoid busy waiting
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
    
    finally:
        elapsed_total = time.time() - start_time
        print("\n" + "=" * 60)
        print("Collection Complete!")
        print(f"Total images collected: {frame_count}")
        print(f"Total time: {elapsed_total:.1f} seconds")
        print(f"Average interval: {elapsed_total/frame_count:.2f} seconds per image" if frame_count > 0 else "")
        print(f"Images saved to: {dataset_folder}")
        print("=" * 60)


if __name__ == "__main__":
    # Configuration
    OUTPUT_FOLDER = "geometry_dash_dataset"  # Folder name on Desktop
    INTERVAL_SECONDS = 2  # Capture every 2 seconds
    MAX_IMAGES = 200  # Maximum images to collect
    
    collect_images(
        output_folder=OUTPUT_FOLDER,
        interval_seconds=INTERVAL_SECONDS,
        max_images=MAX_IMAGES
    )

