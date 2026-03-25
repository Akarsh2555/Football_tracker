import cv2
import numpy as np
import supervision as sv

class Visualizer:
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0, 
                 map_scale: float = 10.0, source_frame_size: tuple = None, invert_y: bool = False):
        """
        Initializes the visualization module.
        
        Args:
            pitch_length_m: Length of pitch in meters.
            pitch_width_m: Width of pitch in meters.
            map_scale: Pixels per meter for the 2D tactical map.
        """
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.map_scale = map_scale
        self.source_frame_size = source_frame_size  # (width_px, height_px)
        self.invert_y = invert_y
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        
        # Trajectories for mapping (dict mapping tracker_id to list of pitch coords)
        self.trajectories = {}
        
        # Pre-render the static pitch map to avoid redrawing every frame
        self._base_pitch_map = self._draw_pitch_map()
        self._draw_legend(self._base_pitch_map)
        
    def _generate_hud_text(self, frame_intel: dict, enriched_positions: list = None) -> list:
        """Helper to generate HUD overlay text from tactics engine output."""
        if not frame_intel:
            return []
            
        possession_prob = frame_intel.get('possession_probability')
        possession_prob_val = possession_prob if possession_prob is not None else 0.0
        
        hud_text = [
            f"Possession: {possession_prob_val:.1%}",
            f"Momentum: {frame_intel.get('momentum_state', 'Neutral')}",
        ]
        
        if frame_intel.get('ball_carrier') is not None:
            carrier_id = frame_intel['ball_carrier']
            ctrl_prob = frame_intel.get('control_probability')
            ctrl_prob_val = ctrl_prob if ctrl_prob is not None else 0.0
            
            carrier_speed_str = ""
            if enriched_positions:
                for p in enriched_positions:
                    if p['player_id'] == carrier_id:
                        speed = p.get('speed', 0.0)
                        carrier_speed_str = f" [{speed:.1f} km/h]"
                        break
                        
            hud_text.append(f"Ball Carrier: {carrier_id}{carrier_speed_str} (Ctrl: {ctrl_prob_val:.2f})")
            
            if frame_intel.get('contextual_xG') is not None:
                hud_text.append(f"Contextual xG (LogReg): {frame_intel['contextual_xG']:.3f}")
            if frame_intel.get('best_pass_EV') is not None:
                hud_text.append(f"Pass EV: {frame_intel['best_pass_EV']:.2f} (Risk: {frame_intel.get('risk_index', 0):.2f})")
                
        return hud_text

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, enriched_positions: list = None, frame_intel: dict = None) -> np.ndarray:
        """
        Draws bounding boxes, IDs, and motion trails on the original frame.
        """
        annotated_frame = frame.copy()
        
        # Filter detections without an assigned tracker ID
        mask = np.array([tr_id is not None for tr_id in detections.tracker_id], dtype=bool)
        tracked_detections = detections[mask]
        
        if len(tracked_detections) == 0:
            return annotated_frame
            
        if enriched_positions:
            # Create a lookup for team info
            team_lookup = {p['player_id']: (p['team_id'], p['is_my_team']) for p in enriched_positions}
            
            labels = []
            for _, _, confidence, class_id, tracker_id, _ in tracked_detections:
                team_id, is_mine = team_lookup.get(tracker_id, (-1, False))
                team_str = "MINE" if is_mine else f"T{team_id}" if team_id != -1 else "UNK"
                labels.append(f"ID:{tracker_id} [{team_str}] {confidence:.2f}")
        else:
            labels = [
                f"ID:{tracker_id} {confidence:.2f}" 
                for _, _, confidence, class_id, tracker_id, _ 
                in tracked_detections
            ]
        
        # Apply annotations
        annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=tracked_detections, labels=labels
        )
        
        # Overlay HUD
        hud_text = self._generate_hud_text(frame_intel, enriched_positions)
        y_offset = 30
        for i, line in enumerate(hud_text):
            cv2.putText(annotated_frame, line, (20, y_offset + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(annotated_frame, line, (20, y_offset + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
        
    def _draw_pitch_map(self) -> np.ndarray:
        """
        Renders a clean 2D football pitch template. Used once during initialization.
        """
        map_w = int(self.pitch_length_m * self.map_scale)
        map_h = int(self.pitch_width_m * self.map_scale)
        
        # Create a green image
        pitch = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        pitch[:] = (60, 150, 60)  # BGR green
        
        line_color = (255, 255, 255)
        thickness = 2
        
        # Pitch outline
        cv2.rectangle(pitch, (0, 0), (map_w, map_h), line_color, thickness)
        
        # Center line
        mid_x = map_w // 2
        cv2.line(pitch, (mid_x, 0), (mid_x, map_h), line_color, thickness)
        
        # Center circle
        center = (mid_x, map_h // 2)
        radius = int(9.15 * self.map_scale)
        cv2.circle(pitch, center, radius, line_color, thickness)
        
        # Penalty areas (simplified representation)
        pen_w = int(16.5 * self.map_scale)
        pen_h = int(40.32 * self.map_scale)
        pen_y = (map_h - pen_h) // 2
        
        cv2.rectangle(pitch, (0, pen_y), (pen_w, pen_y + pen_h), line_color, thickness)
        cv2.rectangle(pitch, (map_w - pen_w, pen_y), (map_w, pen_y + pen_h), line_color, thickness)
        
        return pitch
        
    def _draw_legend(self, pitch: np.ndarray):
        """Draws a permanent color legend in the top right of the pitch."""
        map_w = pitch.shape[1]
        legend_x = map_w - 200
        legend_y = 20
        
        # Draw background panel
        cv2.rectangle(pitch, (legend_x - 10, legend_y - 15), (map_w - 10, legend_y + 110), (40, 40, 40), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        
        cv2.circle(pitch, (legend_x + 10, legend_y + 5), 6, (255, 0, 0), -1)
        cv2.putText(pitch, "My Team", (legend_x + 25, legend_y + 10), font, scale, (255, 255, 255), 1)
        
        cv2.circle(pitch, (legend_x + 10, legend_y + 35), 6, (0, 0, 255), -1)
        cv2.putText(pitch, "Opposition", (legend_x + 25, legend_y + 40), font, scale, (255, 255, 255), 1)
        
        cv2.line(pitch, (legend_x, legend_y + 65), (legend_x + 20, legend_y + 65), (0, 255, 255), 2)
        cv2.putText(pitch, "Best Pass", (legend_x + 25, legend_y + 70), font, scale, (255, 255, 255), 1)
        
        cv2.circle(pitch, (legend_x + 10, legend_y + 95), 8, (255, 255, 255), 2)
        cv2.circle(pitch, (legend_x + 10, legend_y + 95), 6, (0, 0, 0), -1)
        cv2.putText(pitch, "Ball Carrier", (legend_x + 25, legend_y + 100), font, scale, (255, 255, 255), 1)
        
    def render_tactical_map(self, frame_id: int, enriched_positions: list, frame_intel: dict = None) -> np.ndarray:
        """
        Draws the 2D pitch map and plots player positions for the current frame.
        
        Args:
            frame_id: current frame index
            enriched_positions: List of dicts with 'player_id', 'x', 'y', 'team_id', etc.
            frame_intel: Dictionary containing tactical intelligence metrics
        """
        # Copy the pre-rendered static pitch map instead of drawing it from scratch
        pitch = self._base_pitch_map.copy()
        
        ball_carrier_id = frame_intel.get('ball_carrier') if frame_intel else None
        
        # 1. Draw pass connection if best pass exists
        if ball_carrier_id is not None and frame_intel.get('best_pass_option') is not None:
            target_id = frame_intel['best_pass_option']
            
            carrier_pos = None
            target_pos = None
            for p in enriched_positions:
                if p['player_id'] == ball_carrier_id:
                    carrier_pos = self._to_map_coords(p['x'], p['y'])
                elif p['player_id'] == target_id:
                    target_pos = self._to_map_coords(p['x'], p['y'])
                    
            if carrier_pos and target_pos:
                cv2.line(pitch, carrier_pos, target_pos, (0, 255, 255), 2, cv2.LINE_AA) # Yellow pass line
        
        # 2. Draw players
        for player in enriched_positions:
            pid = player['player_id']
            map_x, map_y = self._to_map_coords(player['x'], player['y'])
            
            # Store point in trajectories
            if pid not in self.trajectories:
                self.trajectories[pid] = []
            self.trajectories[pid].append((map_x, map_y))
            
            # Keep only last N points for trajectory trail (e.g., 30 frames)
            if len(self.trajectories[pid]) > 30:
                self.trajectories[pid].pop(0)
                
            # Draw trajectory
            traj_pts = self.trajectories[pid]
            if len(traj_pts) > 1:
                for i in range(1, len(traj_pts)):
                    cv2.line(pitch, traj_pts[i-1], traj_pts[i], (200, 200, 200), 2)
                    
            # Draw player dot
            # Color by team
            if 'team_id' in player and player['team_id'] != -1:
                color = (255, 0, 0) if player.get('is_my_team', False) else (0, 0, 255) # Blue for MY TEAM, Red for opposition
            else:
                color = sv.ColorPalette.DEFAULT.by_idx(pid).as_bgr()
                
            # If this player is the ball carrier, draw a white halo around them
            if pid == ball_carrier_id:
                cv2.circle(pitch, (map_x, map_y), 10, (255, 255, 255), -1) # White halo
                cv2.circle(pitch, (map_x, map_y), 7, (0, 0, 0), -1) # Black boundary
                
            cv2.circle(pitch, (map_x, map_y), 6, color, -1)
            cv2.circle(pitch, (map_x, map_y), 6, (0, 0, 0), 1)
            
            # Draw ID
            cv2.putText(pitch, str(pid), (map_x+8, map_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(pitch, str(pid), (map_x+8, map_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
        # 3. Draw Tactical HUD on Map
        hud_text = self._generate_hud_text(frame_intel, enriched_positions)
        y_offset = 30
        for i, line in enumerate(hud_text):
            # Draw black outline then white text for legibility on the green grass
            cv2.putText(pitch, line, (20, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(pitch, line, (20, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return pitch

    def _to_map_coords(self, x: float, y: float):
        """Convert an input coordinate to map coordinates in pixels."""
        # First, assume coordinates are in pitch meters
        if 0.0 <= x <= self.pitch_length_m and 0.0 <= y <= self.pitch_width_m:
            pitch_x = x
            pitch_y = y
        elif self.source_frame_size is not None:
            frame_w, frame_h = self.source_frame_size
            if frame_w > 0 and frame_h > 0:
                pitch_x = x * self.pitch_length_m / frame_w
                pitch_y = y * self.pitch_width_m / frame_h
            else:
                pitch_x = np.clip(x, 0.0, self.pitch_length_m)
                pitch_y = np.clip(y, 0.0, self.pitch_width_m)
        else:
            pitch_x = np.clip(x, 0.0, self.pitch_length_m)
            pitch_y = np.clip(y, 0.0, self.pitch_width_m)

        if self.invert_y:
            pitch_y = self.pitch_width_m - pitch_y

        map_x = int(np.clip(pitch_x * self.map_scale, 0, self.pitch_length_m * self.map_scale - 1))
        map_y = int(np.clip(pitch_y * self.map_scale, 0, self.pitch_width_m * self.map_scale - 1))
        return map_x, map_y
