import numpy as np
import scipy.spatial.distance as dist

class ExpectedThreatLinear:
    """
    Computes Expected Threat (xT) using linear algebra (matrix inversion) 
    instead of iterative Markov chains.
    Grid size: 16x12 nodes representing the pitch.
    """
    def __init__(self, nx=16, ny=12):
        self.nx = nx
        self.ny = ny
        self.n_states = nx * ny
        
        # Initialize default transition (T), movement probability (M), and shot threat (S)
        # These would ideally be calibrated from events data (e.g. Statsbomb)
        self.T = self._build_transition_matrix()
        self.M = self._build_movement_matrix()
        self.S = self._build_shot_vector()
        
    def _build_transition_matrix(self):
        """
        Builds a basic T matrix defining probability of moving from state i to state j.
        Shapes: (N, N) where N = nx * ny
        Provides a basic heuristic: higher probability moving towards the opponent goal.
        """
        T = np.zeros((self.n_states, self.n_states))
        for x in range(self.nx):
            for y in range(self.ny):
                i = y * self.nx + x
                # Probability of transition to adjacent cells
                total_prob = 0
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                        j = ny_ * self.nx + nx_
                        # Simplistic bias towards attacking (x direction)
                        prob = 1.0 + (dx * 0.5) 
                        T[i, j] = prob
                        total_prob += prob
                
                # Normalize probabilities row-wise
                if total_prob > 0:
                    T[i, :] /= total_prob
        return T

    def _build_movement_matrix(self):
        """
        Diagonal matrix M representing the probability of a player in state i choosing to 
        move/pass rather than shoot.
        """
        M = np.eye(self.n_states)
        for x in range(self.nx):
            for y in range(self.ny):
                i = y * self.nx + x
                # Higher probability of shooting when closer to goal (x = nx-1, mid y)
                dist_to_goal = np.sqrt((self.nx - 1 - x)**2 + (self.ny/2 - y)**2)
                shoot_prob = np.exp(-dist_to_goal / 3.0)
                M[i, i] = 1.0 - shoot_prob
        return M

    def _build_shot_vector(self):
        """
        Shot success probability vector S for each state.
        """
        S = np.zeros(self.n_states)
        for x in range(self.nx):
            for y in range(self.ny):
                i = y * self.nx + x
                dist_to_goal = np.sqrt((self.nx - 1 - x)**2 + (self.ny/2 - y)**2)
                S[i] = 0.5 * np.exp(-dist_to_goal / 2.0)
        return S

    def compute_xt_surface(self) -> np.ndarray:
        """
        Calculates xT surface using: X = (I - MT)^-1 S
        """
        I = np.eye(self.n_states)
        # Solve (I - MT) X = S  --> X = (I - MT)^-1 S
        MT = self.M @ self.T
        try:
            X = np.linalg.solve(I - MT, self.S)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            X = np.linalg.pinv(I - MT) @ self.S
            
        # Reshape to 2D grid
        xt_surface = X.reshape((self.ny, self.nx))
        return xt_surface


class KinematicPitchControl:
    """
    Spearman's Physics-based Pitch Control model calculating Time-To-Intercept (TTI).
    """
    def __init__(self, nx=50, ny=50, max_speed=8.0, reaction_time=0.3):
        self.nx = nx
        self.ny = ny
        self.max_speed = max_speed        # m/s
        self.reaction_time = reaction_time # s
        
        # Grid coordinates
        self.grid_x = np.linspace(0, 105, self.nx)
        self.grid_y = np.linspace(0, 68, self.ny)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)
        self.grid_points = np.column_stack((self.X.ravel(), self.Y.ravel()))

    def calculate_tti(self, player_pos: np.ndarray, player_vel: np.ndarray, target_pos: np.ndarray) -> float:
        """
        Calculate Time To Intercept for a single player to a target position.
        player_pos: [x, y], player_vel: [vx, vy], target_pos: [x, y]
        """
        # Distance to target
        dist = np.linalg.norm(target_pos - player_pos)
        
        # Simplistic kinematic approach: reaction time + time to cover distance at max speed
        # Incorporating current velocity direction (dot product)
        vel_mag = np.linalg.norm(player_vel)
        dir_to_target = (target_pos - player_pos) / (dist + 1e-6)
        
        if vel_mag > 0:
            vel_dir = player_vel / vel_mag
            alignment = np.dot(vel_dir, dir_to_target)
            # Penalize moving in the opposite direction
            speed_penalty = (1 - alignment) * 0.5 
        else:
            speed_penalty = 0

        # Estimate time to reach target
        time_to_reach = self.reaction_time + speed_penalty + (dist / self.max_speed)
        return time_to_reach

    def compute_pitch_control(self, team_a_players: list, team_b_players: list) -> np.ndarray:
        """
        Computes Pitch Control probability surface.
        team_players format: [{'pos': [x, y], 'vel': [vx, vy]}, ...]
        Returns 2D probability array for Team A.
        """
        pc_surface = np.zeros(self.nx * self.ny)
        
        for i, target in enumerate(self.grid_points):
            # Find min TTI for Team A
            min_tti_a = min([self.calculate_tti(np.array(p['pos']), np.array(p['vel']), target) for p in team_a_players]) if team_a_players else 100
            
            # Find min TTI for Team B
            min_tti_b = min([self.calculate_tti(np.array(p['pos']), np.array(p['vel']), target) for p in team_b_players]) if team_b_players else 100
            
            # Logistic function based on difference in arrival time
            # Positive value means Team A arrives sooner
            tti_diff = min_tti_b - min_tti_a
            prob_a = 1 / (1 + np.exp(-tti_diff * 2.0)) # Scale factor 2.0 controls certainty
            pc_surface[i] = prob_a
            
        return pc_surface.reshape((self.ny, self.nx))


def calculate_epv(pc_surface: np.ndarray, xt_surface: np.ndarray) -> np.ndarray:
    """
    Expected Possession Value (EPV) = Pitch Control Probability * Expected Threat
    Note: Ensure surfaces are interpolated to the same resolution before multiplying.
    """
    import cv2
    
    # Resize xT surface to match Pitch Control surface dimensions if different
    if pc_surface.shape != xt_surface.shape:
        xt_surface_resized = cv2.resize(xt_surface, (pc_surface.shape[1], pc_surface.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        xt_surface_resized = xt_surface
        
    epv_surface = pc_surface * xt_surface_resized
    return epv_surface

# Example usage/test
if __name__ == "__main__":
    xt_model = ExpectedThreatLinear(16, 12)
    xt_surf = xt_model.compute_xt_surface()
    
    pc_model = KinematicPitchControl(50, 50)
    # Mock players
    team_a = [{'pos': [20, 30], 'vel': [2, 1]}, {'pos': [60, 40], 'vel': [5, 0]}]
    team_b = [{'pos': [30, 30], 'vel': [-2, 0]}, {'pos': [50, 40], 'vel': [-3, -1]}]
    
    pc_surf = pc_model.compute_pitch_control(team_a, team_b)
    
    epv = calculate_epv(pc_surf, xt_surf)
    print("EPV Surface shape:", epv.shape)
    print("EPV Max value:", np.max(epv))
