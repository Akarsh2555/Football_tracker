import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import supervision as sv

# Ensure ByteTrack works even without Ultralytics running (just directly feeding detections)
from tracker import Tracker

def load_ground_truth(labels_path="mock_data/mock_labels.csv"):
    gt = {}
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            frame_id = int(row[0])
            track_id = int(row[1])
            class_id = int(row[2])
            x1, y1, x2, y2 = map(float, row[3:])
            
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append({
                'id': track_id,
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2]
            })
    return gt

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def generate_noisy_detections(gt_frame, noise_level=0.1, miss_prob=0.05):
    """
    Simulates a noisy YOLO detector.
    - Adds random jitter to bounding boxes
    - Randomly drops some detections
    """
    bboxes = []
    class_ids = []
    confidences = []
    
    for obj in gt_frame:
        if np.random.rand() < miss_prob:
            continue # simulate miss
            
        box = obj['bbox']
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Add noise
        noisy_x1 = box[0] + np.random.randn() * w * noise_level
        noisy_y1 = box[1] + np.random.randn() * h * noise_level
        noisy_x2 = box[2] + np.random.randn() * w * noise_level
        noisy_y2 = box[3] + np.random.randn() * h * noise_level
        
        bboxes.append([noisy_x1, noisy_y1, noisy_x2, noisy_y2])
        class_ids.append(obj['class_id'])
        confidences.append(np.clip(1.0 - np.random.randn() * 0.1, 0.1, 1.0))
        
    if len(bboxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
        
    return np.array(bboxes), np.array(class_ids), np.array(confidences)

def run_evaluation(noise_level=0.0):
    gt = load_ground_truth()
    
    # Initialize Tracker components without actual YOLO model inference
    my_tracker = sv.ByteTrack()
    
    id_switches = 0
    total_matched = 0
    total_gt = 0
    total_false_positives = 0
    total_misses = 0
    
    # Mapping between ground truth IDs and ByteTrack IDs
    gt_to_pred_id = {}
    
    print(f"Running evaluation with noise level {noise_level}...")
    
    for frame_id in sorted(list(gt.keys())):
        gt_frame = gt[frame_id]
        gt_player_frame = [obj for obj in gt_frame if obj['class_id'] == 0]
        total_gt += len(gt_player_frame)
        
        bboxes, class_ids, confidences = generate_noisy_detections(gt_frame, noise_level=noise_level)
        
        if len(bboxes) > 0:
            # Create supervision Detections object
            detections = sv.Detections(
                xyxy=bboxes,
                confidence=confidences,
                class_id=class_ids
            )
            
            # Filter players only matching the real tracker logic
            player_detections = detections[detections.class_id == 0]
            
            if len(player_detections) > 0:
                tracked_players = my_tracker.update_with_detections(player_detections)
            else:
                # empty detections
                tracked_players = sv.Detections.empty()
        else:
            tracked_players = sv.Detections.empty()
            
        # Evaluation: match tracked_players to gt_player_frame
        if len(gt_player_frame) > 0 and len(tracked_players) > 0:
            iou_matrix = np.zeros((len(gt_player_frame), len(tracked_players)))
            
            for i, gt_obj in enumerate(gt_player_frame):
                for j, pred_box in enumerate(tracked_players.xyxy):
                    iou_matrix[i, j] = calculate_iou(gt_obj['bbox'], pred_box)
                    
            # Use Hungarian algorithm for optimal assignment
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_gt = set()
            matched_pred = set()
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > 0.3: # IoU threshold
                    matched_gt.add(r)
                    matched_pred.add(c)
                    total_matched += 1
                    
                    gt_id = gt_player_frame[r]['id']
                    pred_id = tracked_players.tracker_id[c]
                    
                    if gt_id in gt_to_pred_id:
                        if gt_to_pred_id[gt_id] != pred_id:
                            id_switches += 1
                            gt_to_pred_id[gt_id] = pred_id
                    else:
                        gt_to_pred_id[gt_id] = pred_id
                        
            total_misses += len(gt_player_frame) - len(matched_gt)
            total_false_positives += len(tracked_players) - len(matched_pred)
            
        else:
            total_misses += len(gt_player_frame)
            total_false_positives += len(tracked_players)
            
    # Calculate Custom MOTA metric
    mota = 1 - (total_misses + total_false_positives + id_switches) / max(1, total_gt)
    
    print("-" * 40)
    print("TRACKING EVALUATION RESULTS")
    print("-" * 40)
    print(f"Total Frames:        {len(gt)}")
    print(f"Total GT Objects:    {total_gt}")
    print(f"True Positives:      {total_matched}")
    print(f"False Negatives:     {total_misses}")
    print(f"False Positives:     {total_false_positives}")
    print(f"ID Switches:         {id_switches}")
    print(f"MOTA Score:          {mota:.4f} (1.0 is perfect)")
    print("-" * 40)

if __name__ == "__main__":
    print("Baseline Test (No Noise, Perfect Detections):")
    run_evaluation(noise_level=0.0)
    
    print("\nRealistic Test (10% BBox Noise, Some Misses):")
    run_evaluation(noise_level=0.1)
