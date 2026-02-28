import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        """
        Initializes the YOLO model and ByteTrack from supervision.
        
        Args:
            model_path (str): Path to the YOLO weights (e.g., yolov8n.pt, yolov8s.pt).
            conf_threshold (float): Minimum confidence threshold for detections.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Initialize ByteTracker from supervision
        self.tracker = sv.ByteTrack()
        
        # Classes we care about: COCO Person ID is 0, Sports Ball is 32
        self.target_classes = [0, 32]
        
    def process_frame(self, frame: np.ndarray):
        """
        Runs YOLO detection and returns tracked objects using ByteTrack.
        
        Args:
            frame: A single BGR frame from OpenCV.
            
        Returns:
            A tuple containing:
              - Detections object from supervision with tracker IDs
              - Detections for the ball (untracked usually, just detections)
        """
        # Run YOLO on the frame
        # Use predict instead of track to bypass Ultralytics internal GMC bugs (cv2.calcOpticalFlowPyrLK)
        # Supervision sv.ByteTrack handles tracking below anyway.
        results = self.model.predict(frame, verbose=False)
        result = results[0]
        
        # Convert YOLO result to supervision Detections
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter detections by target classes and confidence threshold
        condition = (
            (np.isin(detections.class_id, self.target_classes)) &
            (detections.confidence > self.conf_threshold)
        )
        filtered_detections = detections[condition]
        
        # Split into players and ball
        player_detections = filtered_detections[filtered_detections.class_id == 0]
        ball_detections = filtered_detections[filtered_detections.class_id == 32]
        
        # Update tracker with player detections only
        tracked_players = self.tracker.update_with_detections(player_detections)
        
        return tracked_players, ball_detections
