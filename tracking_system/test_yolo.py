import cv2
from tracker import Tracker

def test_tracker():
    tracker = Tracker()
    # Create a dummy image instead of loading a video
    # 640x360 as reported by the user's video info
    print("Creating dummy frame...")
    dummy_frame = cv2.imread("C:/Users/Akarsh/OneDrive/Desktop/synapse/tracking_system/mock_data/mock_video.mp4") # This won't work, need to read a static frame or create one
    
    import numpy as np
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    print("Testing tracker.process_frame()...")
    tracked_players, ball_detections = tracker.process_frame(dummy_frame)
    print("Tracked Players Length:", len(tracked_players))
    print("Ball Detections Length:", len(ball_detections))
    print("Success!")

if __name__ == "__main__":
    test_tracker()
