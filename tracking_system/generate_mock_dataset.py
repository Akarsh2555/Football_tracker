import os
import cv2
import csv
import numpy as np

def create_mock_dataset(
    output_dir="mock_data",
    video_name="mock_video.mp4",
    labels_name="mock_labels.csv",
    num_frames=100,
    width=640,
    height=480,
    fps=30
):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, video_name)
    labels_path = os.path.join(output_dir, labels_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Define objects: {'id': int, 'pos': [x, y], 'vel': [vx, vy], 'size': [w, h], 'color': (B, G, R), 'class_id': int}
    objects = [
        {'id': 1, 'pos': [50.0, 50.0], 'vel': [4.0, 2.0], 'size': [30, 60], 'color': (255, 0, 0), 'class_id': 0},      # Player 1
        {'id': 2, 'pos': [500.0, 400.0], 'vel': [-3.0, -1.0], 'size': [30, 60], 'color': (0, 255, 0), 'class_id': 0},   # Player 2
        {'id': 3, 'pos': [300.0, 100.0], 'vel': [1.0, 5.0], 'size': [30, 60], 'color': (0, 0, 255), 'class_id': 0},     # Player 3
        {'id': 4, 'pos': [100.0, 300.0], 'vel': [5.0, -2.0], 'size': [30, 60], 'color': (255, 255, 0), 'class_id': 0},  # Player 4
        {'id': 5, 'pos': [200.0, 200.0], 'vel': [6.0, 4.0], 'size': [15, 15], 'color': (255, 255, 255), 'class_id': 32} # Ball
    ]
    
    # Prepare CSV
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'track_id', 'class_id', 'x1', 'y1', 'x2', 'y2'])
        
        for frame_idx in range(num_frames):
            # Create a blank black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            for obj in objects:
                # Update position
                obj['pos'][0] += obj['vel'][0]
                obj['pos'][1] += obj['vel'][1]
                
                # Bounce off walls
                if obj['pos'][0] < 0 or obj['pos'][0] + obj['size'][0] > width:
                    obj['vel'][0] *= -1
                    obj['pos'][0] = max(0, min(obj['pos'][0], width - obj['size'][0]))
                if obj['pos'][1] < 0 or obj['pos'][1] + obj['size'][1] > height:
                    obj['vel'][1] *= -1
                    obj['pos'][1] = max(0, min(obj['pos'][1], height - obj['size'][1]))
                    
                x1 = int(obj['pos'][0])
                y1 = int(obj['pos'][1])
                x2 = int(obj['pos'][0] + obj['size'][0])
                y2 = int(obj['pos'][1] + obj['size'][1])
                
                # Draw object
                cv2.rectangle(frame, (x1, y1), (x2, y2), obj['color'], -1)
                
                # Write ground truth label
                writer.writerow([frame_idx, obj['id'], obj['class_id'], x1, y1, x2, y2])
                
            out_video.write(frame)
            
    out_video.release()
    print(f"Mock dataset generated at {output_dir}")
    print(f"Video: {video_path}")
    print(f"Labels: {labels_path}")

if __name__ == "__main__":
    create_mock_dataset()
