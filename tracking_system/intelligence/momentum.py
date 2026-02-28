from collections import deque
import numpy as np

class TemporalMomentum:
    def __init__(self, window_size_frames=150, alpha=0.1): # alpha for EMA smoothing
        self.window = deque(maxlen=window_size_frames)
        self.current_ema = 0.0
        self.alpha = alpha
        
    def update(self, frame_intel):
        """
        Updates momentum vector using EMA.
        frame_intel: Unified Frame Intelligence dict
        """
        self.window.append(frame_intel)
        
        # Determine the instantaneous un-smoothed momentum payload
        current_xg = frame_intel.get('contextual_xG')
        possession_prob = frame_intel.get('possession_probability', 0.5)
        
        # If we have xg, use it. If not, use possession advantage as a weak proxy, scaled down.
        if current_xg is not None:
            raw_momentum = current_xg
        else:
            # We don't have ball in attacking phase, momentum tends toward 0,
            # or slightly negative if opponent has high possession
            raw_momentum = (possession_prob - 0.5) * 0.1 # Very small payload
            
        # Update EMA
        self.current_ema = (self.alpha * raw_momentum) + ((1.0 - self.alpha) * self.current_ema)
        
        if len(self.window) < 30:
            return "Neutral"
            
        # Classify EMA into states
        if self.current_ema > 0.05:
            return "Positive"
        elif self.current_ema < -0.05:
            return "Negative"
        else:
            return "Neutral"
