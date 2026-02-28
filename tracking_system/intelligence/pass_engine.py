import numpy as np

class PassEngine:
    def __init__(self):
        pass
        
    def evaluate_passes(self, ball_carrier_pos, teammates, opponents, xg_model):
        """
        Evaluates potential passes and returns best pass EV.
        """
        best_pass_pid = None
        best_ev = 0.0
        pass_options = []
        
        for tm in teammates:
            tm_pos = np.array([tm['x'], tm['y']])
            dist = np.linalg.norm(tm_pos - ball_carrier_pos)
            
            # Continuous interception probability heuristic
            interception_risk = 0.0
            for opp in opponents:
                opp_pos = np.array([opp['x'], opp['y']])
                # Distance from opponent to passing lane
                line_vec = tm_pos - ball_carrier_pos
                line_len = np.linalg.norm(line_vec)
                if line_len == 0: continue
                
                line_dir = line_vec / line_len
                opp_vec = opp_pos - ball_carrier_pos
                proj_len = np.dot(opp_vec, line_dir)
                
                if 0 < proj_len < line_len:
                    perp_dist = np.linalg.norm(opp_vec - proj_len * line_dir)
                    # Continuous decay: high risk at 0m, decaying to 0 risk at ~3m
                    if perp_dist < 3.0:
                        risk_contribution = 0.5 * (1.0 - (perp_dist / 3.0))**2
                        interception_risk += risk_contribution
                        
            # Cap interception risk logically, distance penalty
            p_success = max(0.01, 1.0 - (dist / 60.0) - min(0.9, interception_risk))
            
            # receiver xG roughly represents threat
            receiver_xg = xg_model.predict(tm_pos, [np.array([o['x'], o['y']]) for o in opponents])
            
            ev = p_success * receiver_xg
            pass_options.append({'receiver_id': tm['player_id'], 'ev': ev, 'p_success': p_success})
            
            if ev > best_ev:
                best_ev = ev
                best_pass_pid = tm['player_id']
                
        return best_pass_pid, best_ev, pass_options
