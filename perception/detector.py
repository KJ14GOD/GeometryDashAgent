from ultralytics import YOLO
import cv2 as cv


class Detector:
    # def __init__(self, model_path='../models/roboflow_model/weights.pt'):
    def __init__(self, model_path='models/roboflow_model/weights.pt'):
        """
        Initialize detector with trained YOLO model.
        
        Args:
            model_path: Path to YOLO model weights (default: yolo11n.pt)
        """
        print(f"Loading YOLO Model from: {model_path}")
        self.model = YOLO(model_path)
        print("YOLO model loaded!")
    
    def detect(self, frame):
        # YOLO needs BGR (3 channels) - frame should already be BGR if grayscale=False
        # Only convert if grayscale was passed (shouldn't happen, but safety check)
        if len(frame.shape) == 2:  # Grayscale (1 channel)
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        # Frame is now BGR (3 channels) - YOLO can process it
        results = self.model(frame)
        return results

    # def detect_player(self, frame):
    #     results = self.detect_objects(frame)
    #     if len(results == 0):
    #         print("No player found")
    #         return None
        

