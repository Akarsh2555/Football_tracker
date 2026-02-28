import cv2
import numpy as np
import argparse
import json
import os
from video_processor import VideoProcessor

# Globals for storing mouse click coordinates
points = []

def click_event(event, x, y, flags, param):
    """Mouse callback function to record clicks."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            # Draw a circle where the use clicked
            img = param.copy()
            for idx, pt in enumerate(points):
                cv2.circle(img, pt, 5, (0, 0, 255), -1)
                cv2.putText(img, f"P{idx+1}", (pt[0]+10, pt[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Calibration - Click 4 points", img)
            
            if len(points) == 4:
                print("\n4 points selected:")
                for i, p in enumerate(points):
                    print(f"  P{i+1}: {p}")
                print("Press 's' to save and exit, or 'r' to reset, or 'q' to quit without saving.")

def main():
    parser = argparse.ArgumentParser(description="Calibrate Homography using the first frame of a video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="calibration.json", help="Path to save the calibration data")
    
    # Let the user define what rectangle on the pitch they are clicking.
    # By default, let's assume they click the 4 corners of the full pitch (0,0 to 105,68)
    # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right (or ordered polygon)
    # Actually, a common easier shape is the penalty box or half pitch.
    # Default: Full Pitch
    parser.add_argument("--length", type=float, default=105.0, help="Real-world length of the clicked rectangle (meters)")
    parser.add_argument("--width", type=float, default=68.0, help="Real-world width of the clicked rectangle (meters)")
    
    # To map to our dst points:
    # Top-Left (x=0, y=0), Top-Right (x=L, y=0), Bottom-Right (x=L, y=W), Bottom-Left (x=0, y=W)
    # StatsBomb coordinate system: (0,0) is top-left.
    
    args = parser.parse_args()
    
    proc = VideoProcessor(args.input)
    # Get the very first frame
    frame_gen = proc.get_frame()
    try:
        idx, first_frame = next(frame_gen)
    except StopIteration:
        print("Error: Could not read any frames from the video.")
        return
    proc.release()
    
    print("\n" + "="*50)
    print("HOMOGRAPHY CALIBRATION TOOL")
    print("="*50)
    print("Instructions:")
    print("1. A window will open showing the first frame of your video.")
    print("2. Click exactly 4 points to define a rectangle on the pitch.")
    print("3. IMPORTANT: Click the points in this exact order:")
    print("      Point 1: Top-Left")
    print("      Point 2: Top-Right")
    print("      Point 3: Bottom-Right")
    print("      Point 4: Bottom-Left")
    print(f"4. We assume this rectangle in the real world is {args.length}m x {args.width}m.")
    print("   (To calibrate a penalty box: use --length 16.5 --width 40.32)")
    print("="*50 + "\n")
    
    window_name = "Calibration - Click 4 points"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, first_frame)
    cv2.setMouseCallback(window_name, click_event, param=first_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            points.clear()
            cv2.imshow(window_name, first_frame)
            print("Points reset. Click 4 points again.")
            
        elif key == ord('q'):
            print("Exiting without saving.")
            break
            
        elif key == ord('s') and len(points) == 4:
            # Prepare destination points
            # Top-Left, Top-Right, Bottom-Right, Bottom-Left
            dst_pts = [
                [0.0, 0.0],
                [args.length, 0.0],
                [args.length, args.width],
                [0.0, args.width]
            ]
            
            calibration_data = {
                "src_pts": points,
                "dst_pts": dst_pts
            }
            
            with open(args.output, 'w') as f:
                json.dump(calibration_data, f, indent=4)
                
            print(f"\nSUCCESS: Calibration data saved to '{args.output}'.")
            print("You can now run main.py and it will automatically use this calibration for highly accurate mapping.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
