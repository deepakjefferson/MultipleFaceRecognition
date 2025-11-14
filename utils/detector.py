"""
Face Detection Module using DeepFace.

This module provides face detection capabilities using RetinaFace or MTCNN
from the DeepFace library. RetinaFace is preferred for better accuracy.
"""

from typing import List, Tuple, Optional
import numpy as np
from deepface import DeepFace
from config import DETECTOR


class FaceDetector:
    """
    Face detector using DeepFace's RetinaFace or MTCNN.
    
    RetinaFace is preferred as it provides better accuracy and handles
    various lighting conditions and face angles better.
    """
    
    def __init__(self, detector_backend: str = None):
        """
        Initialize the face detector.
        
        Args:
            detector_backend: Detection model to use ("retinaface" or "mtcnn")
                            If None, uses config.DETECTOR
        """
        self.detector_backend = detector_backend or DETECTOR
        
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Detect all faces in a frame.
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
            
        Returns:
            List of face detection dictionaries, each containing:
            - 'facial_area': dict with 'x', 'y', 'w', 'h' keys
            - 'confidence': Detection confidence score (if available)
        """
        try:
            # DeepFace expects RGB format, so convert from BGR
            frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            
            # Use represent() to get face detections with bounding boxes
            # This returns a list of dicts with 'facial_area' and 'embedding'
            detections = DeepFace.represent(
                img_path=frame_rgb,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False  # Don't align here, we just want detection
            )
            
            # If no faces detected, return empty list
            if not detections:
                return []
            
            # Extract detection info (facial_area)
            face_data = []
            for detection in detections:
                if isinstance(detection, dict) and 'facial_area' in detection:
                    face_data.append({
                        'facial_area': detection['facial_area'],
                        'confidence': detection.get('confidence', 1.0)
                    })
            
            return face_data
            
        except Exception as e:
            # If detection fails (e.g., no faces found), return empty list
            return []
    
    def detect_faces_with_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces and return bounding boxes in (x1, y1, x2, y2) format.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        detections = self.detect_faces(frame)
        boxes = []
        
        for detection in detections:
            if 'facial_area' in detection:
                facial_area = detection['facial_area']
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                boxes.append((x, y, x + w, y + h))
        
        return boxes

