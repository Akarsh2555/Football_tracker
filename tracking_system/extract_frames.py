import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, frames_per_sec=1):
    """
    Extracts frames from a video file at a specified rate to use for Roboflow annotations.
    
    Args:
        video_path (str): Path to the source video.
        output_dir (str): Directory to save the extracted frames.
        frames_per_sec (int): How many frames to extract per second of video. 
                              Default is 1 (extracts one frame every second).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate how many frames to skip to achieve the desired frames_per_sec
    # E.g., if video is 30 FPS and we want 1 frame per sec, we extract every 30th frame
    frame_interval = int(fps / frames_per_sec)
    
    if frame_interval < 1:
        frame_interval = 1 
        
    print(f"Video FPS: {fps:.2f}, Total Frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames (approx {frames_per_sec} frames per second).")
    
    extracted_count = 0
    frame_idx = 0
    
    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            extracted_count += 1
            
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    print(f"\nDone! Extracted {extracted_count} frames to: {output_dir}")
    print(f"You can now upload these images to Roboflow for annotation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for Roboflow")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="roboflow_dataset", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=int, default=1, help="Number of frames to extract per second of video (default: 1)")
    
    args = parser.parse_args()
    extract_frames(args.input, args.output_dir, args.fps)
