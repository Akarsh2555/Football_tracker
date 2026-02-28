import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict, deque

class KalmanFilter2D:
    def __init__(self, dt=1.0):
        # State: [x, y, vx, vy]
        self.dt = dt
        self.state = np.zeros(4)
        self.covariance = np.eye(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        
        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise
        self.R = np.eye(2) * 1.0
        
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.state[:2] = measurement
            self.state[2:] = 0
            self.initialized = True
            return self.state[:2], self.state[2:]
            
        # Predict
        predicted_state = self.F @ self.state
        predicted_cov = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update
        y = measurement - (self.H @ predicted_state)
        S = self.H @ predicted_cov @ self.H.T + self.R
        K = predicted_cov @ self.H.T @ np.linalg.inv(S)
        
        self.state = predicted_state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ predicted_cov
        
        return self.state[:2], self.state[2:]


class SpatialEngine:
    def __init__(self, fps=30.0, team_colors_k=2):
        self.fps = fps
        self.dt = 1.0 / fps
        self.filters = {}  # tracker_id -> KalmanFilter2D
        self.history = defaultdict(lambda: deque(maxlen=5)) # to compute acceleration
        
        # Team Classification
        self.team_colors_k = team_colors_k
        self.kmeans = None
        self.team_color_samples = []
        self.max_samples_for_kmeans = 300
        self.my_team_label = 0  # Can be configured later
        self.team_assignments = {} # tracker_id -> team_id
        
    def _extract_dominant_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        # Crop the middle 50% of the bounding box to avoid grass and focus on jersey
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return None
            
        cx, cy = w // 2, h // 2
        crop_w, crop_h = int(w * 0.5), int(h * 0.5)
        
        c_x1 = max(0, cx - crop_w // 2)
        c_y1 = max(0, cy - crop_h // 2)
        c_x2 = min(w, cx + crop_w // 2)
        c_y2 = min(h, cy + crop_h // 2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
            
        center_crop = crop[c_y1:c_y2, c_x1:c_x2]
        if center_crop.size == 0:
            center_crop = crop
            
        # Convert to HSV
        hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV ignoring very dark or very bright pixels (like shadows or numbers)
        mask = cv2.inRange(hsv_crop, np.array([0, 30, 30]), np.array([180, 255, 255]))
        mean_color = cv2.mean(hsv_crop, mask=mask)[:3]
        
        if sum(mean_color) == 0:
            # Fallback if mask is empty
            mean_color = cv2.mean(hsv_crop)[:3]
            
        return np.array(mean_color)

    def process_frame(self, frame_idx, frame, raw_player_positions):
        """
        Enriches raw player positions with smoothed coordinates, kinematics, and team classification.
        Args:
            frame_idx: current frame index
            frame: pixel frame array
            raw_player_positions: list of dicts with 'player_id', 'x', 'y', 'bbox'
        Returns:
            list of enriched spatial objects
        """
        enriched_positions = []
        
        # 1. Team Classification (Collect colors until kmeans is fit)
        if self.kmeans is None:
            for p in raw_player_positions:
                if len(self.team_color_samples) < self.max_samples_for_kmeans:
                    color = self._extract_dominant_color(frame, p['bbox'])
                    if color is not None:
                        self.team_color_samples.append(color)
            
            if len(self.team_color_samples) >= self.max_samples_for_kmeans:
                # Fit KMeans
                self.kmeans = KMeans(n_clusters=self.team_colors_k, random_state=42, n_init=10)
                self.kmeans.fit(self.team_color_samples)
                print("Team classification KMeans initialized.")
                
        for p in raw_player_positions:
            pid = p['player_id']
            curr_pos = np.array([p['x'], p['y']])
            
            # --- Kinematics (Kalman Filter) ---
            if pid not in self.filters:
                self.filters[pid] = KalmanFilter2D(dt=self.dt)
            
            smoothed_pos, velocity = self.filters[pid].update(curr_pos)
            
            # Compute Speed
            vx, vy = velocity
            speed_ms = np.linalg.norm(velocity)
            speed_kmh = speed_ms * 3.6

            
            # Compute Acceleration via historical velocity
            hist = self.history[pid]
            if len(hist) > 0:
                prev_v = hist[-1]['v']
                ax = (vx - prev_v[0]) / self.dt
                ay = (vy - prev_v[1]) / self.dt
            else:
                ax, ay = 0.0, 0.0
                
            acceleration = np.linalg.norm([ax, ay])
            
            # Update history
            hist.append({'pos': smoothed_pos, 'v': velocity})
            
            # --- Team Classification ---
            team_id = -1
            if self.kmeans is not None:
                if pid in self.team_assignments:
                    team_id = self.team_assignments[pid]
                else:
                    color = self._extract_dominant_color(frame, p['bbox'])
                    if color is not None:
                        # Predict cluster
                        team_id = self.kmeans.predict([color])[0]
                        self.team_assignments[pid] = team_id
            
            # Pack enriched fields
            enriched = {
                'frame_id': frame_idx,
                'player_id': pid,
                'team_id': team_id,
                'is_my_team': (team_id == self.my_team_label) if team_id != -1 else False,
                'raw_x': p['x'],
                'raw_y': p['y'],
                'x': smoothed_pos[0],
                'y': smoothed_pos[1],
                'vx': vx,
                'vy': vy,
                'speed': speed_kmh,
                'acceleration': acceleration,
                'bbox': p['bbox']
            }
            enriched_positions.append(enriched)
            
        return enriched_positions
