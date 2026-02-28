import numpy as np

class RiskModel:
    def __init__(self):
        self.defensive_line_x = 20.0 # Approximate
        
    def compute_risk_index(self, pass_options, ball_carrier_pos, opponents):
        """
        Computes Decision Risk Index for Passes.
        Risk = Interception Prob * Counterattack Threat
        """
        if not pass_options:
            return 0.0
            
        best_pass = max(pass_options, key=lambda x: x['ev'])
        p_intercept = 1.0 - best_pass['p_success']
        
        # Counterattack threat heuristic: how many opponents are ahead of the ball carrier?
        opps_ahead = sum(1 for o in opponents if o['x'] < ball_carrier_pos[0]) # assuming we attack right (increasing x)
        threat = min(1.0, opps_ahead / 5.0)
        
        risk_index = p_intercept * threat
        return risk_index
