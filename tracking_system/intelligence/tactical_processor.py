import cv2
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict, Optional, Union
from collections import deque

class TacticalProcessor:
    """
    Core Intelligence Engine for Synapse AI.
    Handles spatial conversions, trajectory smoothing, team clustering, and advanced tactical analytics.
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, fps: float = 25.0):
        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width
        self.fps = fps
        self.homography_matrix: Optional[np.ndarray] = None
        
        # Color clustering
        from sklearn.cluster import KMeans
        self.kmeans: Optional[KMeans] = None
        self.team_color_samples: List[np.ndarray] = []
        
        # Trajectory history for Savitzky-Golay smoothing: {player_id: [(x, y), ...]}
        self.trajectory_history: Dict[int, deque] = {}
        self.window_length = 5
        self.polyorder = 2
        
        # Expected Threat (xT) Grid Setup (16 length x 12 width)
        self.grid_l = 16
        self.grid_w = 12
        self.xt_grid = self._initialize_xt_grid()

    # =========================================================================
    # PHASE 1: CV & SPATIAL PIPELINE
    # =========================================================================

    def calibrate_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """
        Computes the Homography matrix using RANSAC for maximum stability against outliers.
        """
        if len(src_pts) >= 4 and len(dst_pts) >= 4:
            self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return True
        return False

    def transform_bbox_to_pitch(self, bbox: List[float]) -> Optional[Tuple[float, float]]:
        """
        Maps a player's bounding box to the 2D pitch using the bottom-center point (the feet),
        and applies boundary constraints to prevent coordinate drifting out of bounds.
        """
        if self.homography_matrix is None:
            return None
            
        x1, y1, x2, y2 = bbox
        feet_x = (x1 + x2) / 2.0
        feet_y = float(y2)
        
        pt_arr = np.array([[[feet_x, feet_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_arr, self.homography_matrix)
        mapped_x, mapped_y = transformed[0][0]
        
        # Boundary constraints snapping
        mapped_x = float(np.clip(mapped_x, 0.0, self.PITCH_LENGTH))
        mapped_y = float(np.clip(mapped_y, 0.0, self.PITCH_WIDTH))
        
        return (mapped_x, mapped_y)

    def extract_and_cluster_team(self, frame: np.ndarray, bbox: List[float]) -> Optional[int]:
        """
        Masks the pitch, converts player crop to HSV, computes dominant color,
        and assigns a deterministic Team ID based on Hue.
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None
            
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Mask out grass (green hue ~35-85)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv_crop, lower_green, upper_green)
        player_mask = cv2.bitwise_not(grass_mask)
        
        # Calculate mean color of the jersey (ignoring grass)
        mean_hsv = cv2.mean(hsv_crop, mask=player_mask)[:3]
        
        if sum(mean_hsv) == 0:
            mean_hsv = cv2.mean(hsv_crop)[:3] # Fallback
            
        color = np.array(mean_hsv)
        
        from sklearn.cluster import KMeans
        # Accumulate samples for initial fitting
        if self.kmeans is None:
            self.team_color_samples.append(color)
            if len(self.team_color_samples) >= 20:
                self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                self.kmeans.fit(self.team_color_samples)
                
                # Deterministic Sorting by Hue (Index 0 in HSV)
                order = np.argsort(self.kmeans.cluster_centers_[:, 0])
                self.cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
            return -1 # Not ready yet
            
        # Predict once fitted
        raw_pred = self.kmeans.predict([color])[0]
        return self.cluster_mapping.get(raw_pred, -1)

    def smooth_trajectory(self, player_id: int, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Applies a Savitzky-Golay filter across a rolling window of historical coordinates
        to remove YOLO bounding box jitter.
        """
        if player_id not in self.trajectory_history:
            self.trajectory_history[player_id] = deque(maxlen=self.window_length)
            
        history = self.trajectory_history[player_id]
        history.append(current_pos)
        
        if len(history) < self.window_length:
            return current_pos # Not enough data to smooth yet
            
        # Extract x and y series
        xs = np.array([p[0] for p in history])
        ys = np.array([p[1] for p in history])
        
        smoothed_x = savgol_filter(xs, self.window_length, self.polyorder)
        smoothed_y = savgol_filter(ys, self.window_length, self.polyorder)
        
        return (float(smoothed_x[-1]), float(smoothed_y[-1]))

    # =========================================================================
    # PHASE 2: TACTICAL GEOMETRY & ANALYTICS
    # =========================================================================

    def _initialize_xt_grid(self) -> np.ndarray:
        """
        Generates a 16x12 Expected Threat (xT) baseline heuristic grid.
        Values increase closer to the opponent's goal (X = 105).
        """
        grid = np.zeros((self.grid_w, self.grid_l))
        for y in range(self.grid_w):
            for x in range(self.grid_l):
                # Basic gradient: threat increases as X approaches 16.
                # Central zones (y closer to center) get a slight boost.
                center_proximity = 1.0 - (abs(y - (self.grid_w / 2.0)) / (self.grid_w / 2.0))
                forward_progression = (x / float(self.grid_l - 1)) ** 2
                
                grid[y, x] = 0.05 + (forward_progression * 0.8) + (center_proximity * 0.15)
        return grid

    def get_xt_value(self, x: float, y: float) -> float:
        """Looks up the Expected Threat heuristic for a specific coordinate."""
        grid_x = int(np.clip(x / (self.PITCH_LENGTH / self.grid_l), 0, self.grid_l - 1))
        grid_y = int(np.clip(y / (self.PITCH_WIDTH / self.grid_w), 0, self.grid_w - 1))
        return float(self.xt_grid[grid_y, grid_x])

    def evaluate_pass_lane(self, passer: Tuple[float, float], receiver: Tuple[float, float], defenders: List[Tuple[float, float]]) -> float:
        """
        Evaluates the viability of a pass based on intercepting defenders using Cross Product geometry.
        Returns a score from 0.0 (blocked) to 1.0 (perfectly clear), modulated by the receiver's xT zone.
        """
        P = np.array(passer)
        R = np.array(receiver)
        PR = R - P
        dist_PR = np.linalg.norm(PR)
        
        if dist_PR < 1.0:
            return 0.0 # Trivial pass to self
            
        interception_risk = 0.0
        
        for d_pos in defenders:
            D = np.array(d_pos)
            PD = D - P
            
            # Projection of D onto the line segment PR
            t = np.dot(PD, PR) / (dist_PR ** 2)
            
            # Defender is completely behind passer or beyond receiver
            if t < 0.0 or t > 1.0:
                continue
                
            # Cross product (2D): Perpendicular distance from D to line segment PR
            # |PR x PD| / |PR|
            cross_prod_2d = abs(PR[0] * PD[1] - PR[1] * PD[0])
            perp_dist = cross_prod_2d / dist_PR
            
            # Interception threshold
            if perp_dist < 1.5:
                # Closer to the line = drastically higher penalty
                penalty = 1.0 - (perp_dist / 1.5)
                interception_risk = max(interception_risk, penalty)
                
        # Base completion probability drops with distance
        base_prob = max(0.01, 1.0 - (dist_PR * 0.012))
        clearance = 1.0 - interception_risk
        
        pass_prob = base_prob * clearance
        
        # Multiply by the receiver's Expected Threat (xT) zone value
        xt_multiplier = self.get_xt_value(*receiver)
        return pass_prob * xt_multiplier

    def detect_defensive_lapse(self, defenders: List[Tuple[float, float]], ball_pos: Tuple[float, float]) -> Optional[int]:
        """
        Calculates the Center of Mass (CoM) of the defense. If the ball is in the defensive third,
        flags the ID of any defender ('ghosting') who strays > 15m from the CoM.
        """
        if len(defenders) < 3:
            return None # Not enough data for structural analysis
            
        # Is the ball in the defensive third? (Assuming X=0 is defensive goal)
        if ball_pos[0] > (self.PITCH_LENGTH / 3.0):
            return None
            
        def_arr = np.array(defenders)
        com_x = np.mean(def_arr[:, 0])
        com_y = np.mean(def_arr[:, 1])
        com = np.array([com_x, com_y])
        
        for idx, d_pos in enumerate(defenders):
            dist_to_com = np.linalg.norm(np.array(d_pos) - com)
            # 15 meter heuristic threshold for a structural lapse
            if dist_to_com > 15.0:
                return idx # ID of the offending defender
                
        return None
