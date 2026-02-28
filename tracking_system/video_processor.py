import cv2
import os

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str = None):
        """
        Initializes the VideoProcessor.
        
        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to the output video file (optional).
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # Open the input video
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video stream or file: {self.input_path}")
            
        # Extract metadata
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video writer if an output path is provided
        self.out = None
        if self.output_path:
            # Using mp4v codec for .mp4 output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
            
    def get_frame(self):
        """
        Generator that yields frames from the input video sequentially.
        """
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
            
    def write_frame(self, frame):
        """
        Writes a single frame to the output video.
        """
        if self.out is not None:
            self.out.write(frame)
            
    def release(self):
        """
        Releases video capture and writer objects.
        """
        self.cap.release()
        if self.out is not None:
            self.out.release()
            
    def get_metadata(self):
        """
        Returns a dictionary containing video metadata.
        """
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames
        }
