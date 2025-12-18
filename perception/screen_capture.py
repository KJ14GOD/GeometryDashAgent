"""
Screen Capture Module for Geometry Dash RL Agent
Captures frames from the game window at a controlled frame rate.
Provides preprocessing capabilities (resize, normalize).
"""
from mss import mss

import time
import cv2 as cv
import numpy as np
from typing import Tuple, Optional
# from detector import Detector # old imports
# from feature_extractor import FeatureExtractor
from perception.detector import Detector
from perception.feature_extractor import FeatureExtractor


class ScreenCapture:
    """
    Captures frames for a specified monitor region
    
    Args:
        monitor_region: Dictionary with 'top', 'left', 'width', 'height'
        target_fps: Target frames per second (default: 60)
        resize: Optional tuple (width, height) to resize frames
        normalize: If True, normalize pixel values to [0, 1] range
    """
    
    def __init__(self, monitor_region: dict, target_fps: int = 60, resize: Optional[Tuple[int, int]] = None, normalize: bool = False, grayscale: bool = False):
        self.monitor = monitor_region
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.resize = resize
        self.normalize = normalize
        self.grayscale = grayscale
        self.sct = mss()
        
    def capture_frame(self) -> np.ndarray:
        start_time = time.time()
        # Capture frame
        sct_img = self.sct.grab(self.monitor)
        img = np.array(sct_img)
        # Convert directly to grayscale if requested (faster than OpenCV conversion)
        if self.grayscale:
            img = (0.114 * img[:, :, 0].astype(np.float32) +  # Blue
                   0.587 * img[:, :, 1].astype(np.float32) +  # Green
                   0.299 * img[:, :, 2].astype(np.float32))    # Red
            img = img.astype(np.uint8)  # Convert back to uint8
        
        # Control frame rate
        elapsed = time.time() - start_time
        sleep_time = max(0, self.frame_time - elapsed) # make sure to capture only 60 fps not more not less
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return img
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Convert BGRA to BGR (OpenCV standard) - only if not already grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        
        # Resize if specified
        if self.resize is not None:
            frame = cv.resize(frame, self.resize)
        
        # Normalize if specified
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def capture_and_preprocess(self) -> np.ndarray:
        frame = self.capture_frame()
        return self.preprocess(frame)
    
    def get_fps(self) -> float:
        return 1.0 / self.frame_time


if __name__ == "__main__":

    print("Starting in 2 seconds...")
    time.sleep(2)

    
    detector = Detector()
    
    monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
    
    feature_extractor = FeatureExtractor(monitor['width'], monitor['height'])
    
    screen_capture = ScreenCapture(
        monitor_region=monitor,
        target_fps=60,
        resize=None,  # Keep original size for now
        normalize=False,  # Keep 0-255 range for visualization
        grayscale=False  # Capture in COLOR for YOLO (needs color information!)
    )
    
    # Setup video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(
        'yolo_detections.mp4',
        fourcc,
        30.0,  # FPS
        (monitor['width'], monitor['height'])  # Frame size
    )
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture and preprocess frame
            frame = screen_capture.capture_and_preprocess()

            results = detector.detect(frame)

            # Frame is already BGR (3 channels) if grayscale=False
            display_frame = frame.copy()

            print("Number of boxes", len(results[0].boxes))

            # Extract player and obstacles using FeatureExtractor
            player_pos, obstacles_ahead, normalized_features = feature_extractor.extract(results)
            
            # Visualization: Draw bounding boxes and centers
            class_names_dict = results[0].names
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x_int = int((x1+x2)/2)
                center_y_int = int((y1+y2)/2)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())                
                class_name = class_names_dict[class_id]  

                if class_id == 0:
                    color = (255, 0, 0)  # block - Blue
                elif class_id == 1:
                    color = (0, 255, 255)  # coin - Yellow
                elif class_id == 2:
                    color = (255, 255, 0)  # platform - Cyan
                elif class_id == 3:
                    color = (0, 0, 255)  # player - Red
                elif class_id == 4:
                    color = (255, 0, 255)  # portal - Magenta
                elif class_id == 5:
                    color = (0, 165, 255)  # spaceship - Orange
                elif class_id == 6:
                    color = (0, 255, 0)  # spike - Green
              
        
                cv.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}:{confidence:.2f}"
                cv.putText(display_frame, label, (x1, max(y1-10, 10)),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw center point with circle
                if class_id == 3:  # player - larger, bright yellow circle
                    cv.circle(display_frame, (center_x_int, center_y_int), 8, (0, 255, 255), -1)
                    pos_text = f"({center_x_int}, {center_y_int})"
                    cv.putText(display_frame, pos_text, (center_x_int - 40, center_y_int + 25),
                              cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:  # obstacles - smaller cyan circles
                    cv.circle(display_frame, (center_x_int, center_y_int), 5, (255, 255, 0), -1)
                    pos_text = f"({center_x_int}, {center_y_int})"
                    cv.putText(display_frame, pos_text, (center_x_int - 30, center_y_int + 20),
                              cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            
            # Print summary every 60 frames
            if frame_count % 60 == 0 and player_pos is not None:
                player_norm = normalized_features["player"]
                env_norm = normalized_features["environment"]
                state_vector = normalized_features["state_vector"]
                print(f"\nFrame {frame_count}: Player at {player_pos} "
                      f"(norm_y={player_norm['player_y_norm']:.2f}, "
                      f"vel_y={player_norm['player_velocity_y_norm']:.2f}, "
                      f"on_ground={player_norm['on_ground']:.0f})")
                print("  Env distances -> "
                      f"ground={env_norm['ground_distance_norm']:.2f}, "
                      f"next_spike={env_norm['next_spike_distance_norm']:.2f}, "
                      f"next_platform={env_norm['next_platform_distance_norm']:.2f}")
                print(f"  State vector (len={len(state_vector)}): {state_vector}")
                print(f"  Found {len(obstacles_ahead)} obstacles ahead:")

                for i, ((dx, dy, dist, cls, _), norm_vals) in enumerate(
                        zip(obstacles_ahead, normalized_features["obstacles"])):
                    print(
                        f"    {i+1}. {cls}: dx={dx:.0f} (norm={norm_vals['dx_norm']:.2f}), "
                        f"dy={dy:.0f} (norm={norm_vals['dy_norm']:.2f}), "
                        f"dist={dist:.0f} (norm={norm_vals['distance_norm']:.2f}), "
                        f"type_norm={norm_vals['class_type_norm']:.2f}, "
                        f"land={norm_vals['can_land_on_top']:.0f}, "
                        f"risk={norm_vals['collision_risk']:.2f}"
                    )
                    
            # Add info text
            info_text = f"Frame: {frame_count} | Detections: {len(results[0].boxes)}"
            cv.putText(display_frame, info_text, (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to video (must be BGR format)
            video_writer.write(display_frame)
            
            frame_count += 1
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Frames: {frame_count} | FPS: {actual_fps:.2f} | Detections: {len(results[0].boxes)}")
            
            # Check for quit
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Release video writer
        video_writer.release()
        cv.destroyAllWindows()
        print("Screen capture test complete")

