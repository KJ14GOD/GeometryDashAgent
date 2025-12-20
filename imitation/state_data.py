'''
This script is used to extract the state vectors from the frames captured during gameplay. Essentially
offline processing of the frames to not mess up the real time processing.
'''

from perception.feature_extractor import FeatureExtractor
from perception.detector import Detector
import numpy as np
import cv2 as cv

class StateData:
    def __init__(self):
        monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
        self.feature_extractor = FeatureExtractor(monitor['width'], monitor['height'])
        self.detector = Detector()  
        self.state_array = []
        self.data = np.load('data/imitation_data.npz')
        self.actions = np.array(self.data['actions'])
        self.action_array = []
        
    
    def extract_states(self, image):
        results = self.detector.detect(image)
        player_pos, obstacles_ahead, normalized_features = self.feature_extractor.extract(results)
        state_vector = normalized_features["state_vector"]
        self.state_array.append(state_vector)

    
    def save_states(self):
        np.savez(
            'data/imitation_state_and_action_data.npz',
            state_array=np.array(self.state_array),
            action_array=np.array(self.action_array),
        )
        print(f"Saved {len(self.state_array)} states and {len(self.action_array)} actions")


if __name__ == "__main__":
    state_data = StateData()
    print("Starting")
    for i, action in enumerate(state_data.actions):
        frame = cv.imread(f'data/expert_frames/frame_{i}.jpg')
        if frame is None:
            print(f"Frame {i} not found")
            continue
        state_data.extract_states(frame)
        if i % 100 == 0:
            print(f"Extracted {i} states")
        state_data.action_array.append(action)
        
    state_data.save_states()
