import numpy as np
import sys
import os

# Ensure the parent directory is in the path to import xg_contextual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from intelligence.xg_contextual import ContextualxGModel
except ImportError as e:
    print(f"Error importing ContextualxGModel: {e}")
    sys.exit(1)

def test_predictions():
    print("--- Testing Contextual xG Model Inference ---")
    model = ContextualxGModel(model_path="intelligence/xg_model.pkl")

    # The pitch is assumed 105x68. The goal is at [105, 34].
    
    # Scenario 1: Close range, centered, no defenders
    player_pos1 = np.array([100.0, 34.0]) # 5m straight in front of goal
    defenders1 = []
    xg1 = model.predict(player_pos1, defenders1)
    print(f"\nScenario 1 - Close range, centered, no defenders:")
    print(f"  Player Pos: {player_pos1}")
    print(f"  xG: {xg1:.4f}")

    # Scenario 2: Close range, centered, 1 defender nearby
    player_pos2 = np.array([100.0, 34.0])
    defenders2 = [np.array([101.0, 34.5])] # Defender 1m away, slightly off-center
    xg2 = model.predict(player_pos2, defenders2)
    print(f"\nScenario 2 - Close range, centered, with pressure:")
    print(f"  Player Pos: {player_pos2}")
    print(f"  Defenders: {defenders2}")
    print(f"  xG: {xg2:.4f}")

    # Scenario 3: Outside the box, sharp angle
    player_pos3 = np.array([90.0, 15.0]) # 15m away, wide angle
    defenders3 = []
    xg3 = model.predict(player_pos3, defenders3)
    print(f"\nScenario 3 - Far away, sharp angle, no defenders:")
    print(f"  Player Pos: {player_pos3}")
    print(f"  xG: {xg3:.4f}")

if __name__ == "__main__":
    test_predictions()
