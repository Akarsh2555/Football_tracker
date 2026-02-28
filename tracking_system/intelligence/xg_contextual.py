import os
import math
import numpy as np
import pickle

class ContextualxGModel:
    def __init__(self, model_path="intelligence/xg_model.pkl"):
        # We assume the tactical engine pitch is typically 105x68
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.goal_pos_local = np.array([self.pitch_length, self.pitch_width / 2.0])
        self.goal_width = 7.32 # Standard goal width
        
        # StatsBomb training coordinates
        self.sb_length = 120.0
        self.sb_width = 80.0
        self.sb_goal = np.array([120.0, 40.0])
        self.sb_goal_width = 8.0 # approximate scale in SB coordinates
        
        # Load the ML Model
        self.model = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Successfully loaded trained ML xG model from {model_path}")
            except Exception as e:
                print(f"Failed to load xG model: {e}. Falling back to heuristic.")
        else:
            print(f"xG Model not found at {model_path}. Falling back to heuristic.")

    def _convert_to_sb_coords(self, player_pos):
        """Converts local [0-105, 0-68] to StatsBomb [0-120, 0-80]"""
        x_scaled = (player_pos[0] / self.pitch_length) * self.sb_length
        y_scaled = (player_pos[1] / self.pitch_width) * self.sb_width
        return np.array([x_scaled, y_scaled])

    def _calculate_sb_angle(self, sb_pos):
        # Returns visible angle of the StatsBomb goal from scaled position in radians
        x, y = sb_pos[0], sb_pos[1]
        post1 = np.array([self.sb_length, 40.0 - (self.sb_goal_width/2)])
        post2 = np.array([self.sb_length, 40.0 + (self.sb_goal_width/2)])
        
        v1 = post1 - sb_pos
        v2 = post2 - sb_pos
        
        dot = np.dot(v1, v2)
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.atan2(det, dot)
        return abs(angle)
        
    def predict(self, player_pos, defenders_pos):
        """
        Computes Contextual xG. Focuses heavily on the true ML model.
        player_pos: [x, y] of shooter
        defenders_pos: list of [x, y] of defenders
        """
        # Convert local coordinates to scaled SB coordinates for ML consistency
        sb_pos = self._convert_to_sb_coords(player_pos)
        
        dist_to_goal = np.linalg.norm(sb_pos - self.sb_goal)
        angle_rad = self._calculate_sb_angle(sb_pos)
        
        base_xg = 0.05
        
        if self.model is not None:
            # Predict using Logistic Regression Model
            # Model expects Features: ['distance', 'angle']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                features = np.array([[dist_to_goal, angle_rad]])
                try:
                    # predict_proba returns [prob_miss, prob_goal]
                    base_xg = self.model.predict_proba(features)[0][1]
                except Exception as e:
                    print(f"ML Prediction failed: {e}")
                    base_xg = self._heuristic_fallback(dist_to_goal, angle_rad)
        else:
            base_xg = self._heuristic_fallback(dist_to_goal, angle_rad)
            
        # Contextual Pressure penalty (ML model didn't train on defenders natively, so we apply heuristically)
        pressure_penalty = 0.0
        nearest_def = float('inf')
        
        for d in defenders_pos:
            diff = player_pos - d
            dist = np.linalg.norm(diff)
            if dist < nearest_def:
                nearest_def = dist
            if dist < 2.5: # 2.5 meters
                pressure_penalty += 0.15 * (1.0 - (dist/2.5))
                
        if nearest_def < 1.0: # Block likelihood is high
            pressure_penalty += 0.15 
            
        # Final contextualized xG
        contextual_xg = max(0.01, base_xg - pressure_penalty)
        return contextual_xg
        
    def _heuristic_fallback(self, sb_distance, sb_angle):
        # Base xG based on angle and non-linear distance prioritizing close range
        base_xg = (sb_angle / (math.pi/2)) * 0.5 + (max(0, 40 - sb_distance) / 40.0) * 0.5
        return np.clip(base_xg, 0.01, 0.95)
