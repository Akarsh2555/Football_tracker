from collections import deque
import numpy as np
from scipy.spatial import ConvexHull

class TeamMetrics:
    def __init__(self, window_size=300): # 10 seconds rolling window for possession
        self.possession_window = deque(maxlen=window_size)
        
    def compute(self, my_team_players, ball_carrier_team, contested):
        """
        Computes compactness, width spread, and rolling possession prob.
        """
        if ball_carrier_team == "MINE" and not contested:
            self.possession_window.append(1)
        elif ball_carrier_team == "OPP" and not contested:
            self.possession_window.append(0)
        else:
            # If contested or UNK, maintain the previous ratio by not adding 1 or 0
            # or add a fractional value. For simplicity, we can ignore contested frames 
            # to let clear possession dictate the rolling metric.
            pass
            
        possession_prob = np.mean(self.possession_window) if len(self.possession_window) > 0 else 0.5
        
        if len(my_team_players) < 3:
            return {
                'possession_prob': possession_prob,
                'team_centroid': None,
                'compactness': 0.0,
                'width_spread': 0.0
            }
            
        pts = np.array([[p['x'], p['y']] for p in my_team_players])
        centroid = np.mean(pts, axis=0)
        
        try:
            hull = ConvexHull(pts)
            compactness = hull.volume # For 2D, volume is area
        except Exception:
            compactness = 0.0
            
        width_spread = np.max(pts[:, 1]) - np.min(pts[:, 1])
        
        return {
            'possession_prob': possession_prob,
            'team_centroid': centroid.tolist(),
            'compactness': compactness,
            'width_spread': width_spread
        }
