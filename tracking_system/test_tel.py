import json
import numpy as np
from intelligence.core import TacticalIntelligenceEngine

def test_intelligence_engine():
    print("=== Testing Tactical Intelligence Engine (TEL) ===")
    
    # 1. Initialize the engine (Team 0 is 'MY TEAM')
    engine = TacticalIntelligenceEngine(my_team_label=0)
    print("\n[+] Engine Initialized Successfully!")
    
    # 2. Setup mock 'enriched_positions' (Output of SpatialEngine)
    enriched_positions = [
        # My Team
        {'player_id': 1, 'x': 50.0, 'y': 34.0, 'team_id': 0, 'is_my_team': True, 'speed': 2.5, 'heading': [1.0, 0.0]},
        {'player_id': 2, 'x': 70.0, 'y': 20.0, 'team_id': 0, 'is_my_team': True, 'speed': 3.0, 'heading': [0.8, 0.6]},
        {'player_id': 3, 'x': 60.0, 'y': 50.0, 'team_id': 0, 'is_my_team': True, 'speed': 1.0, 'heading': [1.0, 0.0]},
        # Opponents
        {'player_id': 4, 'x': 55.0, 'y': 33.0, 'team_id': 1, 'is_my_team': False, 'speed': 2.0, 'heading': [-1.0, 0.0]},
        {'player_id': 5, 'x': 65.0, 'y': 40.0, 'team_id': 1, 'is_my_team': False, 'speed': 1.5, 'heading': [-0.5, 0.5]},
        {'player_id': 6, 'x': 80.0, 'y': 34.0, 'team_id': 1, 'is_my_team': False, 'speed': 0.0, 'heading': [0.0, 0.0]}
    ]
    
    # 3. Setup mock ball position
    # The ball is currently near player 1 (My Team)
    ball_pitch_pos = [51.0, 34.0] 
    
    print("\n[+] Testing Frame Processing...")
    frame_idx = 100
    
    # 4. Process frame through the intelligence engine
    intel_output = engine.process_frame(frame_idx, enriched_positions, ball_pitch_pos)
    
    # 5. Output Results
    print(f"\n--- Frame {frame_idx} Tactical Intelligence ---")
    print(json.dumps(intel_output, indent=4))
    
    # Assertions / Validations
    print("\n[+] Validating Logic...")
    if intel_output['ball_carrier'] == 1:
        print("  - [PASS] Correctly identified Player 1 as ball carrier.")
    else:
        print(f"  - [FAIL] Expected Player 1 as ball carrier, got {intel_output['ball_carrier']}")
        
    if intel_output['contextual_xG'] is not None:
        print(f"  - [PASS] Computed Contextual xG: {intel_output['contextual_xG']:.3f}")
    else:
        print("  - [FAIL] Failed to compute Contextual xG.")
        
    if intel_output['best_pass_option'] is not None:
        print(f"  - [PASS] Computed Best Pass Option: Player {intel_output['best_pass_option']} (EV: {intel_output['best_pass_EV']:.3f})")
    else:
        print("  - [FAIL] Failed to compute Best Pass Option.")
        
    if intel_output['risk_index'] is not None:
        print(f"  - [PASS] Computed Risk Index: {intel_output['risk_index']:.3f}")
    else:
        print("  - [FAIL] Failed to compute Risk Index.")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_intelligence_engine()
