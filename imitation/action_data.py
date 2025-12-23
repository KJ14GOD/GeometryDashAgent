from perception.screen_capture import ScreenCapture
from pynput import keyboard
import numpy as np
import cv2 as cv
import time 
import os


class ActionData:
    def __init__(self):
        monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
        self.screen_capture = ScreenCapture(monitor_region=monitor, target_fps=60, resize=None, normalize=False, grayscale=False)
        self.action_array = []
        self.timestamps = []
        self.space_pressed = False 
        self.should_stop = False
        self.frame_count = 0
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start() # start the thread for the keyboard listener
        # Create frames folder
        self.frames_folder = 'data/expert_frames'
        os.makedirs(self.frames_folder, exist_ok=True)
       
    def extract_data(self):
        timestamp = time.time()
        frame = self.screen_capture.capture_and_preprocess()        
        
        if self.space_pressed:
            self.action_array.append(1) # jumped
        else:
            self.action_array.append(0) # did not jump
        
        self.timestamps.append(timestamp)
        
        # Save frame (same frame used for state extraction = 1-to-1 match)
        filename = f"frame_{self.frame_count}.jpg"
        filepath = os.path.join(self.frames_folder, filename)
        cv.imwrite(filepath, frame)
        
        self.frame_count += 1

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = True
        elif key == keyboard.Key.esc:
            self.should_stop = True
            print("ESC pressed - stopping data collection...")
    
    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = False

if __name__ == "__main__":
    action_data = ActionData()
    print("Starting in 3 seconds (saving frames to data/expert_frames/)")
    time.sleep(3)
    
    start_time = time.time()
    try:
        for i in range(10000):
            if action_data.should_stop:
                break
            action_data.extract_data()
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                actual_fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"Extracted {i} frames | FPS: {actual_fps:.1f}")
    
    except KeyboardInterrupt:
        print("stopped early")
    
    finally:
        action_data.listener.stop()

    # Save both actions and timestamps
    np.savez('data/imitation_data.npz', actions=np.array(action_data.action_array), timestamps=np.array(action_data.timestamps))
    
    # Calculate final statistics
    total_time = action_data.timestamps[-1] - action_data.timestamps[0] if len(action_data.timestamps) > 1 else 0
    avg_fps = len(action_data.action_array) / total_time if total_time > 0 else 0
    
    print(f"Saved {len(action_data.action_array)} actions and timestamps")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total duration: {total_time:.1f} seconds")