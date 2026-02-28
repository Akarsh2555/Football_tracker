import numpy as np

class BallControlModel:
    def __init__(self, max_speed=10.0, reaction_time=0.2, temperature=1.5):
        self.max_speed = max_speed
        self.reaction_time = reaction_time
        self.temperature = temperature # Temperature for softmax scaling
        
    def compute_control_probabilities(self, ball_pos, players, ball_detections=None):
        """
        ball_pos: [x, y] pitch coordinates
        players: list of dicts with 'player_id', 'x', 'y', 'team_id', 'speed', 'bbox'
        ball_detections: supervision Detections object for the raw YOLO ball (if available)
        Returns: {player_id: prob}, ball_carrier_id, contested
        """
        # 1. PIXEL GEOMETRY (HIGH PRECISION)
        # Check if we have raw pixel bounding boxes to evaluate true visual intersection
        if ball_detections is not None and len(ball_detections) > 0 and len(players) > 0 and all('bbox' in p for p in players):
            b_box = ball_detections.xyxy[0]
            ball_c = np.array([(b_box[0] + b_box[2]) / 2.0, (b_box[1] + b_box[3]) / 2.0])
            
            distances = {}
            for p in players:
                p_box = p['bbox']
                # Approximate feet location (bottom center of player box)
                feet_c = np.array([(p_box[0] + p_box[2]) / 2.0, p_box[3]])
                pixel_dist = np.linalg.norm(ball_c - feet_c)
                
                # Normalize pixel_dist by the player's bounding box height to make it zoom/scale invariant
                p_height = p_box[3] - p_box[1]
                norm_dist = pixel_dist / (p_height + 1e-6)
                distances[p['player_id']] = norm_dist
                
            probs = {}
            min_d = min(distances.values())
            
            # If the ball is extremely far from everyone visually (e.g. > 2.5x player heights away) it's a loose ball
            if min_d > 2.5:
                # Nobody cleanly possesses the ball, output zeros
                return {p['player_id']: 0.0 for p in players}, None, False
                
            # Tighter Temperature (0.4) for sharp cutoffs on who possesses the ball based on pixel proximity
            temp = 0.4
            exp_sum = sum(np.exp(-(d - min_d) / temp) for d in distances.values())
            
            best_prob = 0
            ball_carrier = None
            
            for pid, d in distances.items():
                prob = np.exp(-(d - min_d) / temp) / exp_sum
                probs[pid] = prob
                if prob > best_prob:
                    best_prob = prob
                    ball_carrier = pid
                    
            contested = best_prob < 0.65
            return probs, ball_carrier, contested

        # 2. HOMOGRAPHY FALLBACK (LOW PRECISION)
        # If raw detections are missing, fallback to distorted 2D Pitch Coordinates
        times_to_ball = {}
        for p in players:
            dist = np.linalg.norm(np.array([p['x'], p['y']]) - ball_pos)
            # Time to intercept
            t = (dist / self.max_speed) + self.reaction_time
            times_to_ball[p['player_id']] = t
            
        # Softmax over negative times, scaled by temperature
        probs = {}
        if not times_to_ball:
            return probs, None, False
            
        # to prevent overflow and scale by temp
        min_t = min(times_to_ball.values())
        exp_sum = sum(np.exp(-(t - min_t) / self.temperature) for t in times_to_ball.values())
        
        best_prob = 0
        ball_carrier = None
        
        for pid, t in times_to_ball.items():
            prob = np.exp(-(t - min_t) / self.temperature) / exp_sum
            probs[pid] = prob
            if prob > best_prob:
                best_prob = prob
                ball_carrier = pid
                
        # Contested if the leader doesn't have a huge advantage
        contested = best_prob < 0.55 
        
        return probs, ball_carrier, contested
