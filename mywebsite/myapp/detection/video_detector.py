"""
Video detection module for Smart Fruit
Handles fruit detection in video files and real-time streams
"""
import cv2
import numpy as np
from pathlib import Path
import time


class VideoFruitDetector:
    """
    Class for detecting fruits in video files and streams
    """
    
    def __init__(self, detector):
        """
        Initialize video detector
        
        Args:
            detector: FruitDetector instance for frame processing
        """
        self.detector = detector
        self.fps = 0
        self.total_frames = 0
        self.processed_frames = 0
        
    def detect_in_video(self, video_path, output_path=None, process_every_n_frames=1):
        """
        Detect fruits in a video file
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video (optional)
            process_every_n_frames: Process every Nth frame for performance
            
        Returns:
            dict: Detection results including frame counts and detections
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % process_every_n_frames == 0:
                # TODO: Implement actual detection on frame
                # detections = self.detector.detect_frame(frame)
                detections = []
                
                all_detections.append({
                    'frame': frame_count,
                    'timestamp': frame_count / self.fps,
                    'detections': detections
                })
                
                # Draw detections on frame
                frame = self.draw_detections_on_frame(frame, detections)
                self.processed_frames += 1
            
            # Write frame to output video
            if out:
                out.write(frame)
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'fps': self.fps,
            'detections': all_detections,
            'output_path': output_path
        }
    
    def detect_in_webcam(self, camera_index=0, duration=None):
        """
        Detect fruits from webcam stream
        
        Args:
            camera_index: Camera device index (default 0)
            duration: Duration in seconds to capture (None for continuous)
            
        Yields:
            tuple: (frame, detections) for each processed frame
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open camera {camera_index}")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check duration limit
            if duration and (time.time() - start_time) > duration:
                break
            
            # TODO: Implement actual detection on frame
            # detections = self.detector.detect_frame(frame)
            detections = []
            
            # Draw detections
            frame = self.draw_detections_on_frame(frame, detections)
            
            yield frame, detections
        
        cap.release()
    
    def draw_detections_on_frame(self, frame, detections):
        """
        Draw bounding boxes and labels on a video frame
        
        Args:
            frame: Video frame (numpy array)
            detections: List of detection dictionaries
            
        Returns:
            numpy array: Annotated frame
        """
        for detection in detections:
            # Extract detection info
            bbox = detection.get('bbox', [])
            label = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def extract_frames(self, video_path, output_dir, interval=1.0):
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            interval: Time interval in seconds between frames
            
        Returns:
            list: Paths to extracted frame images
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return frame_paths
    
    def get_video_info(self, video_path):
        """
        Get information about a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            dict: Video information
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
