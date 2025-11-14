"""
Face Embedding Module using DeepFace.

This module generates face embeddings (feature vectors) using deep learning
models like Facenet512 or ArcFace. These embeddings are used for face recognition.
"""

from typing import List, Tuple, Optional
import numpy as np
from deepface import DeepFace
from config import MODEL_NAME, DETECTOR


class FaceEmbedder:
    """
    Face embedder using DeepFace models.
    
    Generates high-dimensional feature vectors (embeddings) from face images.
    These embeddings can be compared using cosine similarity for recognition.
    """
    
    def __init__(self, model_name: str = None, detector_backend: str = None):
        """
        Initialize the face embedder.
        
        Args:
            model_name: Model to use for embeddings ("Facenet512", "ArcFace", etc.)
                       If None, uses config.MODEL_NAME
            detector_backend: Detection model to use ("retinaface" or "mtcnn")
                            If None, uses config.DETECTOR
        """
        self.model_name = model_name or MODEL_NAME
        self.detector_backend = detector_backend or DETECTOR
        
    def extract_embedding(self, frame: np.ndarray, face_box: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding from a frame.
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
            face_box: Optional bounding box (x1, y1, x2, y2) to extract specific face
                     If None, automatically detects the face
                     
        Returns:
            Face embedding as numpy array, or None if face not found
        """
        try:
            # Convert BGR to RGB for DeepFace
            frame_rgb = frame[:, :, ::-1]
            
            # If face_box is provided, crop the face region
            if face_box is not None:
                x1, y1, x2, y2 = face_box
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_rgb.shape[1], x2)
                y2 = min(frame_rgb.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    return None
                    
                frame_rgb = frame_rgb[y1:y2, x1:x2]
            
            # Generate embedding using DeepFace
            # represent() returns a list of dicts with 'embedding' and 'facial_area'
            embedding_obj = DeepFace.represent(
                img_path=frame_rgb,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Extract embedding vector
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embedding = embedding_obj[0]['embedding']
                return np.array(embedding)
            elif isinstance(embedding_obj, dict) and 'embedding' in embedding_obj:
                return np.array(embedding_obj['embedding'])
            
            return None
            
        except Exception as e:
            # If embedding extraction fails, return None
            return None
    
    def extract_embeddings_batch(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for multiple faces in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            face_boxes: List of bounding boxes (x1, y1, x2, y2) for each face
            
        Returns:
            List of embeddings (one per face box), None if extraction failed
        """
        embeddings = []
        for box in face_boxes:
            embedding = self.extract_embedding(frame, box)
            embeddings.append(embedding)
        return embeddings

