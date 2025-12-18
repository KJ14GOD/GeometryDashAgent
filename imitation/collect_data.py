from perception.feature_extractor import FeatureExtractor
from perception.screen_capture import ScreenCapture
from perception.detector import Detector
from pynput import keyboard
import numpy as np
import time 


class CollectData:
    def __init__(self):
        monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
        self.feature_extractor = FeatureExtractor(monitor['width'], monitor['height'])
        self.screen_capture = ScreenCapture(monitor_region=monitor, target_fps=60, resize=None, normalize=False, grayscale=False)
        self.detector = Detector()
        self.action_array = []
        self.state_array = []
        self.space_pressed = False
        self.should_stop = False
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start() # start the thread for the keyboard listener
       
    def extract_data(self):
        frame = self.screen_capture.capture_and_preprocess()
        results = self.detector.detect(frame)
        player_pos, obstacles_ahead, normalized_features = self.feature_extractor.extract(results)

        state_vector = normalized_features["state_vector"]
        self.state_array.append(state_vector) # append the state vector to the array 
        
        if self.space_pressed:
            self.action_array.append(1) # jumped
        else:
            self.action_array.append(0) # did not jump

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
    collect_data = CollectData()
    print("Starting in 3 seconds")
    time.sleep(3)
    try:
        for i in range(10000):
            if collect_data.should_stop:
                break
            collect_data.extract_data()
            
            if i % 100 == 0:
                print(f"Extracted {i} frames")
    
    except KeyboardInterrupt:
        print("stopped early")
    
    
    finally:
        collect_data.listener.stop()

    
    np.savez('data/expert_data.npz', state_array=np.array(collect_data.state_array), action_array=np.array(collect_data.action_array))
    print(f"Saved {len(collect_data.state_array)} frames")