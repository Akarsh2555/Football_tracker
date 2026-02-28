import os
import json
import pandas as pd

class DataExporter:
    def __init__(self, output_dir: str):
        """
        Initializes the data exporter.
        
        Args:
            output_dir: Path to directory where CSV and JSON will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data = []
        self.intel_data = []
        
    def add_frame_data(self, frame_idx: int, player_positions: list, frame_intel: dict = None):
        """
        Adds tracking data for a single frame.
        
        Args:
            frame_idx: The current frame number.
            player_positions: List of dicts: {'player_id': id, 'x': x_m, 'y': y_m}
        """
        for pos in player_positions:
            data_row = {
                'frame': frame_idx,
                'player_id': pos['player_id'],
                'pitch_x_meter': pos['x'],
                'pitch_y_meter': pos['y']
            }
            # Add spatial kinematics if available
            if 'vx' in pos:
                data_row.update({
                    'raw_x': pos.get('raw_x', pos['x']),
                    'raw_y': pos.get('raw_y', pos['y']),
                    'vx': pos['vx'],
                    'vy': pos['vy'],
                    'speed': pos['speed'],
                    'acceleration': pos['acceleration'],
                    'team_id': pos['team_id'],
                    'is_my_team': pos['is_my_team']
                })
                
            if frame_intel:
                data_row.update({
                    'is_ball_carrier': pos['player_id'] == frame_intel.get('ball_carrier'),
                    'control_probability': frame_intel.get('control_probability') if pos['player_id'] == frame_intel.get('ball_carrier') else None
                })
                
            self.data.append(data_row)
            
        # We can also store frame-level intelligence in a separate list
        if not hasattr(self, 'intel_data'):
            self.intel_data = []
        if frame_intel:
            self.intel_data.append(frame_intel)
            
    def export(self, file_prefix: str = "tracking_data"):
        """
        Exports the accumulated data to CSV and JSON formats.
        """
        if not self.data:
            print("No data collected to export.")
            return
            
        df = pd.DataFrame(self.data)
        
        # Sort values chronologically and by player
        df = df.sort_values(by=['frame', 'player_id'])
        
        # Export to CSV
        csv_path = os.path.join(self.output_dir, f"{file_prefix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Data exported to {csv_path}")
        
        if hasattr(self, 'intel_data') and self.intel_data:
            intel_df = pd.DataFrame(self.intel_data)
            intel_csv_path = os.path.join(self.output_dir, f"{file_prefix}_intel.csv")
            intel_df.to_csv(intel_csv_path, index=False)
            print(f"Intel data exported to {intel_csv_path}")
        
        # Process for structured JSON
        # Create a nested dictionary structure for JSON
        # Output format:
        # [
        #   {
        #     "frame": 0,
        #     "players": [{"id": 1, "x": 10.5, "y": 20.3}, ...]
        #   }, ...
        # ]
        json_data = []
        grouped = df.groupby('frame')
        for frame, group in grouped:
            frame_data = {
                "frame": int(frame),
                "players": []
            }
            for _, row in group.iterrows():
                p_data = {
                    "id": int(row['player_id']),
                    "x": float(row['pitch_x_meter']),
                    "y": float(row['pitch_y_meter'])
                }
                if 'vx' in row and not pd.isna(row['vx']):
                    p_data.update({
                        "vx": float(row['vx']),
                        "vy": float(row['vy']),
                        "speed": float(row['speed']),
                        "acceleration": float(row['acceleration']),
                        "team_id": int(row['team_id']),
                        "is_my_team": bool(row['is_my_team'])
                    })
                frame_data["players"].append(p_data)
            json_data.append(frame_data)
            
        json_path = os.path.join(self.output_dir, f"{file_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Data exported to {json_path}")
