import os
import math
import numpy as np
import pandas as pd
import pickle
from statsbombpy import sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# 1. Configuration
# Competition 43 = FIFA World Cup, Season 106 = 2022
COMPETITION_ID = 43
SEASON_ID = 106
OUTPUT_DIR = "intelligence"

# Standard Statsbomb pitch is 120 x 80. The goal is centered on the y-axis (y=40) at x=120.
# The standard goal width is ~7.32 meters (approx 8 yards).
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_POS = np.array([120.0, 40.0])
GOAL_WIDTH_Y = 8.0 # approximate scale in SB coordinates

def calculate_distance(player_loc):
    """Euclidean distance to the center of the goal."""
    return np.linalg.norm(np.array(player_loc) - GOAL_POS)

def calculate_angle(player_loc):
    """
    Visible angle of the goal mouth from the shooter's position.
    Returns angle in radians.
    """
    x, y = player_loc[0], player_loc[1]
    
    # Coordinates of the two goal posts
    post1 = np.array([PITCH_LENGTH, 40.0 - (GOAL_WIDTH_Y / 2)])
    post2 = np.array([PITCH_LENGTH, 40.0 + (GOAL_WIDTH_Y / 2)])
    
    player_vec = np.array([x, y])
    
    v1 = post1 - player_vec
    v2 = post2 - player_vec
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle shots directly on the goal line to avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return math.pi
        
    cos_theta = dot_product / (norm_v1 * norm_v2)
    # Clip to avoid floating point errors causing acos domain issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return math.acos(cos_theta)

def fetch_and_prepare_data():
    print(f"Fetching matches for Competition {COMPETITION_ID}, Season {SEASON_ID}...")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    match_ids = matches['match_id'].tolist()
    
    print(f"Found {len(match_ids)} matches. Harvesting shot data...")
    
    all_shots = []
    
    for mid in match_ids:
        try:
            events = sb.events(match_id=mid)
            # Filter only shots
            if 'type' in events.columns and 'shot_outcome' in events.columns:
                shots = events[events['type'] == 'Shot']
                
                # We only want Open Play shots for a standard tracking model (no penalties/free kicks)
                if 'play_pattern' in shots.columns:
                    shots = shots[shots['play_pattern'] == 'Regular Play']
                
                # Extract necessary columns: location [x, y] and outcome
                for _, row in shots.iterrows():
                    loc = row['location']
                    outcome = row['shot_outcome']
                    
                    if isinstance(loc, list) and len(loc) >= 2:
                        dist = calculate_distance(loc)
                        angle = calculate_angle(loc)
                        
                        # Labels: Goal = 1, everything else (Saved, Off T, Blocked, Post) = 0
                        is_goal = 1 if outcome == 'Goal' else 0
                        
                        all_shots.append({
                            'distance': dist,
                            'angle': angle,
                            'is_goal': is_goal
                        })
        except Exception as e:
            print(f"Failed to process match {mid}: {e}")
            continue

    df = pd.DataFrame(all_shots)
    print(f"\nSuccessfully extracted {len(df)} open-play shots.")
    return df

def train_model(df):
    if len(df) == 0:
        print("Error: No valid shots found to train on.")
        return
        
    print("\n--- Model Training ---")
    
    # Features (X) and Target (y)
    X = df[['distance', 'angle']]
    y = df['is_goal']
    
    print(f"Goals: {sum(y)}, Non-Goals: {len(y) - sum(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Logistic Regression
    # We use 'balanced' class_weight because goals are rare compared to misses
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\nEvaluation Metrics (Test Set n={len(X_test)}):")
    print(f"Accuracy:         {accuracy_score(y_test, y_pred):.4f}")
    if len(set(y_test)) > 1:
        print(f"ROC-AUC:          {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Brier Score Loss: {brier_score_loss(y_test, y_prob):.4f} (Lower = Better curve fit)")
    
    print("\nFeature Coefficients (Distance, Angle):")
    print([round(c, 4) for c in model.coef_[0]])
    print("Intercept:")
    print(round(model.intercept_[0], 4))
    
    # Save the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "xg_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"\nModel successfully saved to {model_path}!")

if __name__ == "__main__":
    df = fetch_and_prepare_data()
    train_model(df)
