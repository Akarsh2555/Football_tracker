import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from collections import deque
import math

class SpatialEngine:
    """
    Handles tracking filtering, trajectory smoothing, velocity/acceleration calculations,
    and unsupervised team color clustering using KMeans on dominant jersey RGB.
    """
    def __init__(self, fps=25.0, team_colors_k=2, smooth_window=5, smooth_poly=2,
                 pitch_length_m=105.0, pitch_width_m=68.0, frame_size=None):
        self.fps = fps
        self.team_colors_k = team_colors_k
        self.my_team_label = 0
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.frame_size = frame_size  # (width_px, height_px)
        
        # History for velocity and acceleration calculations
        self.position_history = {}  # {player_id: deque of (x, y)}
        self.velocity_history = {}  # {player_id: deque of speed}
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        
        # Color clustering
        self.kmeans = None
        self.team_color_samples = []
        self.team_assignments = {}
        self.cluster_mapping = {}
        
    def _extract_dominant_color(self, frame, bbox):
        """
        Extracts the dominant jersey color using raw RGB averaging.
        Crops the center 50% of the bounding box to focus on the jersey torso,
        avoiding peripheral background pixels.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return None
            
        # Crop center 50% of bounding box (jersey torso region)
        crop_h_start = y1 + int(h * 0.25)
        crop_h_end = y1 + int(h * 0.75)
        crop_w_start = x1 + int(w * 0.25)
        crop_w_end = x1 + int(w * 0.75)
        
        # Bounds check
        crop_h_start = max(0, crop_h_start)
        crop_h_end = min(frame.shape[0], crop_h_end)
        crop_w_start = max(0, crop_w_start)
        crop_w_end = min(frame.shape[1], crop_w_end)
        
        crop = frame[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        if crop.size == 0:
            return None
            
        # Mask out grass pixels using HSV green range
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv_crop, lower_green, upper_green)
        player_mask = cv2.bitwise_not(grass_mask)
        
        # Count non-grass pixels
        non_grass_count = cv2.countNonZero(player_mask)
        if non_grass_count < 10:
            # Too few non-grass pixels, use full crop
            mean_color = cv2.mean(crop)[:3]
        else:
            # Use only non-grass pixels for color extraction (raw RGB/BGR)
            mean_color = cv2.mean(crop, mask=player_mask)[:3]
        
        return np.array(mean_color)
    
    def _smooth_position(self, player_id, raw_x, raw_y):
        """
        Applies Savitzky-Golay smoothing across a rolling window of historical coordinates
        to reduce YOLO bounding box jitter.
        """
        if player_id not in self.position_history:
            self.position_history[player_id] = deque(maxlen=self.smooth_window)
        
        history = self.position_history[player_id]
        history.append((raw_x, raw_y))
        
        if len(history) < self.smooth_window:
            return raw_x, raw_y
            
        xs = np.array([p[0] for p in history])
        ys = np.array([p[1] for p in history])
        
        try:
            smoothed_x = savgol_filter(xs, self.smooth_window, self.smooth_poly)
            smoothed_y = savgol_filter(ys, self.smooth_window, self.smooth_poly)
            return float(smoothed_x[-1]), float(smoothed_y[-1])
        except Exception:
            return raw_x, raw_y

    def _to_pitch_coords(self, x, y):
        """Convert incoming coordinate system into pitch meters as needed."""
        if 0.0 <= x <= self.pitch_length_m and 0.0 <= y <= self.pitch_width_m:
            return x, y

        if self.frame_size is not None:
            frame_w, frame_h = self.frame_size
            if frame_w > 0 and frame_h > 0:
                x = x * self.pitch_length_m / frame_w
                y = y * self.pitch_width_m / frame_h

        x = max(0.0, min(x, self.pitch_length_m))
        y = max(0.0, min(y, self.pitch_width_m))
        return x, y

    def process_frame(self, frame_idx, frame, raw_player_positions):
        """
        Processes a single frame's worth of player positions.
        
        Args:
            frame_idx: Current frame index
            frame: The raw BGR OpenCV frame (used for color extraction)
            raw_player_positions: List of dicts with 'player_id', 'x', 'y', 'bbox'
            
        Returns:
            List of enriched player dicts with team_id, is_my_team, vx, vy, speed, acceleration
        """
        dt = 1.0 / self.fps
        enriched = []
        
        # 1. Team Classification - Collect color samples until KMeans can fit
        if self.kmeans is None:
            for p in raw_player_positions:
                color = self._extract_dominant_color(frame, p['bbox'])
                if color is not None:
                    self.team_color_samples.append(color)
                    
            # 20 samples ensures both teams are represented for deterministic clustering
            if len(self.team_color_samples) >= 20:
                self.kmeans = KMeans(n_clusters=self.team_colors_k, random_state=42, n_init=10)
                self.kmeans.fit(self.team_color_samples)
                
                # Sort clusters by HSV Value (Brightness) for deterministic team assignment.
                # This perfectly resolves scenarios where teams wear pure White or Black uniforms,
                # which generate random Hue values and would formerly flip team colors.
                centers_bgr = self.kmeans.cluster_centers_
                # Convert each center to HSV to get Value (brightness)
                brightness_values = []
                for center in centers_bgr:
                    bgr_pixel = np.uint8([[center]])
                    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
                    brightness_values.append(hsv_pixel[0, 0, 2])  # V channel
                
                order = np.argsort(brightness_values)
                self.cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
            
        # 2. Process each player
        for p in raw_player_positions:
            pid = p['player_id']
            raw_x, raw_y = p['x'], p['y']
            
            # Apply trajectory smoothing
            px, py = self._smooth_position(pid, raw_x, raw_y)
            px, py = self._to_pitch_coords(px, py)
            
            # Predict team
            team_id = -1
            if self.kmeans is not None:
                if pid in self.team_assignments:
                    team_id = self.team_assignments[pid]
                else:
                    color = self._extract_dominant_color(frame, p['bbox'])
                    if color is not None:
                        raw_team_id = self.kmeans.predict([color])[0]
                        team_id = self.cluster_mapping.get(raw_team_id, -1)
                        if team_id != -1:
                            self.team_assignments[pid] = team_id
                            
            is_my_team = (team_id == self.my_team_label)
            
            # 3. Kinematics (Velocity, Speed, Acceleration)
            if pid not in self.velocity_history:
                self.velocity_history[pid] = deque(maxlen=5)
            
            vx, vy, speed, acceleration = 0.0, 0.0, 0.0, 0.0
            
            if pid in self.position_history and len(self.position_history[pid]) >= 2:
                pts = self.position_history[pid]
                p1 = pts[-2]
                p2 = pts[-1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                vx = dx / dt
                vy = dy / dt
                speed_mps = math.hypot(vx, vy)
                speed = speed_mps * 3.6  # Convert m/s to km/h
                
                # Cap unrealistic speeds (tracking jitter)
                if speed > 40.0:
                    speed = 40.0
                    
                # Acceleration (change in speed over time)
                speed_history = self.velocity_history[pid]
                speed_history.append(speed)
                if len(speed_history) >= 2:
                    acceleration = (speed_history[-1] - speed_history[-2]) / dt
                    # Cap unrealistic acceleration
                    acceleration = max(-50.0, min(50.0, acceleration))
            
            player_data = p.copy()
            player_data.update({
                'x': px,  # Smoothed position
                'y': py,  # Smoothed position
                'raw_x': raw_x,
                'raw_y': raw_y,
                'team_id': team_id,
                'is_my_team': is_my_team,
                'vx': vx,
                'vy': vy,
                'speed': speed,
                'acceleration': acceleration
            })
            enriched.append(player_data)
            
        return enriched
