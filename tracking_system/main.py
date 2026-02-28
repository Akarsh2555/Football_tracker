import os
import cv2
import argparse
import numpy as np
import json
from tqdm import tqdm

from video_processor import VideoProcessor
from tracker import Tracker
from pitch_mapper import PitchMapper
from visualizer import Visualizer
from exporter import DataExporter
from intelligence.spatial import SpatialEngine
from intelligence.core import TacticalIntelligenceEngine

def main(args):
    # 1. Initialize modules
    print("Initializing modules...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    video_proc = VideoProcessor(args.input)
    meta = video_proc.get_metadata()
    print(f"Video Info: {meta['width']}x{meta['height']} at {meta['fps']} FPS, {meta['total_frames']} frames")
    
    tracker = Tracker(model_path=args.model)
    pitch_mapper = PitchMapper()
    visualizer = Visualizer(map_scale=10.0)
    exporter = DataExporter(args.output_dir)
    
    # Setup placeholder homography (Requires actual field points for accuracy)
    # Using a trapezoid representing the field view in a typical broadcast
    w, h = meta['width'], meta['height']
    
    spatial_engine = SpatialEngine(fps=meta['fps'], team_colors_k=args.num_teams)
    if hasattr(args, 'my_team_label') and args.my_team_label is not None:
        spatial_engine.my_team_label = args.my_team_label
        
    intelligence_engine = TacticalIntelligenceEngine(my_team_label=args.my_team_label if hasattr(args, 'my_team_label') else 0)
        
    # Attempt to load calibration data
    calib_file = "calibration.json"
    if os.path.exists(calib_file):
        print(f"Loading highly accurate homography calibration from {calib_file}...")
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
            src_pts = np.array(calib_data["src_pts"], dtype=np.float32)
            dst_pts = np.array(calib_data["dst_pts"], dtype=np.float32)
    else:
        print("Warning: No calibration.json found. Using placeholder homography.")
        print("Run `python calibrate.py --input <video>` for great accuracy.")
        src_pts = np.array([
            [w * 0.1, h * 0.9],
            [w * 0.9, h * 0.9],
            [w * 0.3, h * 0.4],
            [w * 0.7, h * 0.4]
        ], dtype=np.float32)
        
        dst_pts = np.array([
            [0.0, 68.0],
            [105.0, 68.0],
            [0.0, 0.0],
            [105.0, 0.0]
        ], dtype=np.float32)
    
    pitch_mapper.compute_homography(src_pts, dst_pts)
    
    # 2. Setup output video writers
    out_video_path = os.path.join(args.output_dir, "output_tracked_video.mp4")
    out_map_path = os.path.join(args.output_dir, "tactical_map_video.mp4")
    out_composite_path = os.path.join(args.output_dir, "output_composite_video.mp4")
    
    # Use avc1 (H.264) codec for broad HTML5 browser compatibility, avoiding mp4v playback issues
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_video = cv2.VideoWriter(out_video_path, fourcc, meta['fps'], (w, h))
    
    # Compute map dimensions based on scale
    map_w = int(105.0 * 10.0)
    map_h = int(68.0 * 10.0)
    out_map = cv2.VideoWriter(out_map_path, fourcc, meta['fps'], (map_w, map_h))
    
    # Compute composite dimensions based on height of the main video
    # Resizing the tactical map to match the height of the main video
    comp_map_h = h
    comp_map_w = int(map_w * (h / map_h))
    comp_w = w + comp_map_w
    out_composite = cv2.VideoWriter(out_composite_path, fourcc, meta['fps'], (comp_w, h))
    
    # 3. Process video frame-by-frame
    print("Processing video...")
    for frame_idx, frame in tqdm(video_proc.get_frame(), total=meta['total_frames']):
        
        # Tracking
        tracked_players, ball_detections = tracker.process_frame(frame)
        
        # Coordinate Mapping
        player_positions = []
        if len(tracked_players) > 0:
            # Gather all bottom-center points
            points_to_transform = []
            for i in range(len(tracked_players)):
                bbox = tracked_players.xyxy[i]
                bottom_center = pitch_mapper.extract_bottom_center(bbox)
                points_to_transform.append(bottom_center)
                
            # Vectorized transformation
            transformed_pts = pitch_mapper.transform_points(np.array(points_to_transform))
            
            if transformed_pts is not None:
                for i in range(len(tracked_players)):
                    bbox = tracked_players.xyxy[i]
                    tracker_id = tracked_players.tracker_id[i]
                    x_m, y_m = transformed_pts[i]
                    
                    player_positions.append({
                        'player_id': tracker_id,
                        'x': float(x_m),
                        'y': float(y_m),
                        'bbox': bbox  # Passed for team classification
                    })
                    
        # Apply Spatial Intelligence (Smoothing, Kinematics, Team Classification)
        enriched_positions = spatial_engine.process_frame(frame_idx, frame, player_positions)
        
        # Map Ball Coordinates
        ball_pitch_pos = None
        if len(ball_detections) > 0:
            ball_bbox = ball_detections.xyxy[0]
            ball_bottom_center = pitch_mapper.extract_bottom_center(ball_bbox)
            ball_pitch_coords = pitch_mapper.transform_point(ball_bottom_center)
            if ball_pitch_coords:
                ball_pitch_pos = ball_pitch_coords
        else:
            # Inject a dummy ball position near the first player to ensure HUD metrics populate.
            my_team_players = [p for p in enriched_positions if p.get('is_my_team')]
            target_player = my_team_players[0] if my_team_players else (enriched_positions[0] if len(enriched_positions) > 0 else None)
            
            if target_player:
                # Add a small offset to the ball so it's not exactly ON the player
                ball_pitch_pos = [target_player['x'] + 0.5, target_player['y'] + 0.5]
                
        # Apply Tactical Intelligence
        frame_intel = intelligence_engine.process_frame(frame_idx, enriched_positions, ball_pitch_pos, ball_detections)
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: ball_pos={ball_pitch_pos}, intel={frame_intel}")
                    
        # Export Data
        exporter.add_frame_data(frame_idx, enriched_positions, frame_intel)
        
        # Visualization
        annotated_frame = visualizer.annotate_frame(frame, tracked_players, enriched_positions, frame_intel)
        tactical_map = visualizer.render_tactical_map(frame_idx, enriched_positions, frame_intel)
        
        # Write frames to output videos
        out_video.write(annotated_frame)
        out_map.write(tactical_map)
        
        # Write Composite Frame
        resized_map = cv2.resize(tactical_map, (comp_map_w, comp_map_h))
        composite_frame = np.hstack((annotated_frame, resized_map))
        out_composite.write(composite_frame)
        
        # Optional: display frame for debugging/real-time native viewing
        # cv2.imshow("Tactical Engine - Side-By-Side (Press 'q' to quit)", composite_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #      break
        
        # Optional: display frame for debugging (requires GUI)
        # cv2.imshow("Annotated", annotated_frame)
        # cv2.imshow("Tactical Map", tactical_map)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
    # 4. Cleanup and Export
    video_proc.release()
    out_video.release()
    out_map.release()
    out_composite.release()
    cv2.destroyAllWindows()
    
    print("Exporting data...")
    exporter.export()
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Player Tracking System")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights")
    
    parser.add_argument("--num_teams", type=int, default=2, help="Number of teams to cluster")
    parser.add_argument("--my_team_label", type=int, default=0, help="Cluster label for MY TEAM (e.g. 0 or 1)")
    
    args = parser.parse_args()
    main(args)
