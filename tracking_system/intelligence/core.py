import numpy as np
from .xg_contextual import ContextualxGModel
from .ball_control import BallControlModel
from .pass_engine import PassEngine
from .risk_model import RiskModel
from .momentum import TemporalMomentum
from .metrics import TeamMetrics

class TacticalIntelligenceEngine:
    def __init__(self, my_team_label=0):
        self.my_team_label = my_team_label
        self.xg_model = ContextualxGModel()
        self.ball_control = BallControlModel()
        self.pass_engine = PassEngine()
        self.risk_model = RiskModel()
        self.momentum = TemporalMomentum()
        self.metrics = TeamMetrics()
        
    def process_frame(self, frame_idx, enriched_positions, ball_pitch_pos, ball_detections=None):
        """
        Processes a single frame and returns a Unified Frame Intelligence Object.
        enriched_positions: output from SpatialEngine
        ball_pitch_pos: [x, y] coordinates of the ball on the pitch (or None)
        ball_detections: raw supervision Detections object for the ball from YOLO
        """
        intel = {
            'frame_id': frame_idx,
            'ball_carrier': None,
            'control_probability': None,
            'contested': False,
            'contextual_xG': None,
            'best_pass_option': None,
            'best_pass_EV': None,
            'risk_index': None,
            'team_compactness': None,
            'possession_probability': None,
            'momentum_state': "Neutral"
        }
        
        # 1. Identify my team and opponents
        my_team = [p for p in enriched_positions if p.get('is_my_team', False)]
        opponents = [p for p in enriched_positions if not p.get('is_my_team', False) and p.get('team_id', -1) != -1]
        
        if ball_pitch_pos is None or len(enriched_positions) == 0:
            return intel # No ball or players, cannot run most tactical models
            
        ball_pos_np = np.array(ball_pitch_pos)
        
        # 2. Ball Control Model (using pixel bounding boxes if available)
        probs, ball_carrier, contested = self.ball_control.compute_control_probabilities(ball_pos_np, enriched_positions, ball_detections)
        intel['ball_carrier'] = ball_carrier
        intel['control_probability'] = probs.get(ball_carrier, 0.0) if ball_carrier is not None else 0.0
        intel['contested'] = contested
        
        # Determine team of ball carrier
        ball_carrier_team = "UNK"
        carrier_pos = None
        for p in enriched_positions:
            if p['player_id'] == ball_carrier:
                ball_carrier_team = "MINE" if p.get('is_my_team', False) else "OPP"
                carrier_pos = np.array([p['x'], p['y']])
                break
                
        # 3. Contextual xG (computed if SOMEONE has the ball)
        if carrier_pos is not None:
            # For demonstration purposes, if ball_carrier_team is not "MINE", 
            # we will pretend they are the attacking team to show the HUD metrics, 
            # or we calculate xG for whoever has the ball vs the others.
            attacking_team = my_team if ball_carrier_team == "MINE" else opponents
            defending_team = opponents if ball_carrier_team == "MINE" else my_team
            
            # Prevent empty lists from crashing models
            if len(attacking_team) == 0:
                attacking_team = [p for p in enriched_positions if p['player_id'] == ball_carrier]
            if len(defending_team) == 0:
                defending_team = [p for p in enriched_positions if p['player_id'] != ball_carrier]

            opp_points = [np.array([o['x'], o['y']]) for o in defending_team]
            
            # Predict xG
            intel['contextual_xG'] = float(self.xg_model.predict(carrier_pos, opp_points))
            
            # 4. Pass Engine & Decision Risk
            best_pass_pid, best_ev, pass_options = self.pass_engine.evaluate_passes(carrier_pos, attacking_team, defending_team, self.xg_model)
            intel['best_pass_option'] = best_pass_pid
            intel['best_pass_EV'] = float(best_ev)
            intel['risk_index'] = float(self.risk_model.compute_risk_index(pass_options, carrier_pos, defending_team))
            
        # 5. Team Metrics
        metrics_res = self.metrics.compute(my_team, ball_carrier_team, contested)
        intel['possession_probability'] = float(metrics_res['possession_prob'])
        intel['team_compactness'] = float(metrics_res['compactness'])
        
        # 6. Temporal Momentum
        intel['momentum_state'] = self.momentum.update(intel)
        
        return intel
