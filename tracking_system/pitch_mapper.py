import cv2
import numpy as np

class PitchMapper:
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0):
        """
        Initializes the PitchMapper to convert pixel coordinates to 2D pitch coordinates.
        
        Args:
            pitch_length_m (float): Length of the pitch in meters.
            pitch_width_m (float): Width of the pitch in meters.
        """
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.homography_matrix = None
        
    def get_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Detects the grass region using HSV color masking.
        
        Args:
            frame: A BGR OpenCV frame.
        Returns:
            A binary mask where the pitch is 255 and background is 0.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return mask
        
    def compute_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray):
        """
        Computes the homography matrix from source image points to destination pitch points.
        
        Args:
            src_pts (np.ndarray): Nx2 array of points in the video frame.
            dst_pts (np.ndarray): Nx2 array of corresponding points on the pitch map (in meters).
        """
        # Ensure at least 4 points are provided
        if len(src_pts) >= 4 and len(dst_pts) >= 4:
            self.homography_matrix, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return self.homography_matrix
        return None
        
    def transform_point(self, point: tuple) -> tuple:
        """
        Transforms a (x, y) pixel coordinate to (x_meter, y_meter) pitch coordinate.
        
        Args:
            point (tuple): (x, y) coordinates in the image.
        Returns:
            Tuple of (x_meter, y_meter) or None if homography implies invalid mapping.
        """
        if self.homography_matrix is None:
            return None
            
        pt_arr = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt_arr, self.homography_matrix)
        
        x_m, y_m = transformed_pt[0][0]
        
        # Optional: clip within pitch dimensions or just return as is
        return (float(x_m), float(y_m))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms an array of (x, y) pixel coordinates to (x_meter, y_meter) pitch coordinates.
        This provides a vectorized alternative to calling transform_point sequentially.
        
        Args:
            points: Nx2 array of (x, y) image coordinates.
        Returns:
            Nx2 array of (x_meter, y_meter) pitch coordinates, or None.
        """
        if self.homography_matrix is None or len(points) == 0:
            return None
            
        pts_arr = np.array([points], dtype=np.float32)
        transformed_pts = cv2.perspectiveTransform(pts_arr, self.homography_matrix)
        
        return transformed_pts[0]
        
    def extract_bottom_center(self, bbox: list) -> tuple:
        """
        Extracts the bottom-center point of a bounding box (x1, y1, x2, y2).
        This point generally represents where the player's feet touch the ground.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        bottom_y = y2
        return (center_x, bottom_y)
