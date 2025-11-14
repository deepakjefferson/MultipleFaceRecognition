#!/usr/bin/env python3
"""
Colab-compatible face recognition script.

This version is adapted for Google Colab environment:
- Processes uploaded images or uses camera widget
- Displays results inline
- Works with Colab's file system
"""

import pickle
from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from deepface import DeepFace
from IPython.display import Image, display
from config import (
    EMBEDDINGS_FILE,
    THRESHOLD,
    COLOR_KNOWN,
    COLOR_UNKNOWN,
    COLOR_TEXT,
    FONT_SCALE,
    FONT_THICKNESS,
    BOX_THICKNESS,
    DETECTOR,
    MODEL_NAME
)
from utils.matcher import FaceMatcher


def load_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from pickle file."""
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded embeddings for {len(embeddings)} people: {list(embeddings.keys())}")
        return embeddings
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_path}' not found!")
        print("Please run train.py first to generate embeddings.")
        return {}
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}


def recognize_faces_in_image(image_path: str, embeddings_path: str = EMBEDDINGS_FILE, 
                             threshold: float = THRESHOLD, output_path: str = "result.jpg"):
    """
    Recognize faces in a single image (Colab-friendly).
    
    Args:
        image_path: Path to input image
        embeddings_path: Path to embeddings.pkl
        threshold: Similarity threshold
        output_path: Path to save result image
        
    Returns:
        List of recognition results
    """
    # Load embeddings
    embeddings = load_embeddings(Path(embeddings_path))
    if not embeddings:
        return []
    
    # Initialize matcher
    matcher = FaceMatcher(embeddings, threshold=threshold)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    # Convert BGR to RGB for DeepFace
    frame_rgb = frame[:, :, ::-1]
    
    results = []
    
    try:
        # Detect and recognize faces
        representations = DeepFace.represent(
            img_path=frame_rgb,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False,
            align=True
        )
        
        # Ensure representations is a list
        if isinstance(representations, dict):
            representations = [representations]
        
        # Process each detected face
        for rep in representations:
            if 'facial_area' in rep and 'embedding' in rep:
                # Extract bounding box
                facial_area = rep['facial_area']
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                bbox = (x, y, x + w, y + h)
                
                # Get embedding
                embedding = np.array(rep['embedding'])
                
                # Match against known faces
                result = matcher.match(embedding)
                
                name = result['name']
                confidence = result['confidence']
                is_match = result['is_match']
                
                # Print result
                if is_match:
                    print(f"Detected: {name}, Confidence: {confidence*100:.0f}%")
                else:
                    print("Unknown person detected")
                
                # Draw on frame
                color = COLOR_KNOWN if is_match else COLOR_UNKNOWN
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
                
                label = f"{name} ({confidence*100:.0f}%)" if is_match else "Unknown"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                )
                
                # Draw text background
                cv2.rectangle(
                    frame,
                    (x, y - text_height - baseline - 10),
                    (x + text_width, y),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    COLOR_TEXT,
                    FONT_THICKNESS
                )
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'is_match': is_match,
                    'bbox': bbox
                })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return []
    
    # Save result
    cv2.imwrite(output_path, frame)
    print(f"\nResult saved to {output_path}")
    
    # Display result in Colab
    display(Image(output_path))
    
    return results


def process_video_file(video_path: str, embeddings_path: str = EMBEDDINGS_FILE,
                       threshold: float = THRESHOLD, output_path: str = "output_video.mp4",
                       max_frames: int = None):
    """
    Process a video file frame by frame (Colab-friendly).
    
    Args:
        video_path: Path to input video
        embeddings_path: Path to embeddings.pkl
        threshold: Similarity threshold
        output_path: Path to save output video
        max_frames: Maximum frames to process (None = all)
    """
    # Load embeddings
    embeddings = load_embeddings(Path(embeddings_path))
    if not embeddings:
        return
    
    # Initialize matcher
    matcher = FaceMatcher(embeddings, threshold=threshold)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_frames = 0
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 2nd frame for speed
        if frame_count % 2 == 0:
            try:
                frame_rgb = frame[:, :, ::-1]
                
                representations = DeepFace.represent(
                    img_path=frame_rgb,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                    align=True
                )
                
                if isinstance(representations, dict):
                    representations = [representations]
                
                # Draw results on frame
                for rep in representations:
                    if 'facial_area' in rep and 'embedding' in rep:
                        facial_area = rep['facial_area']
                        x = facial_area['x']
                        y = facial_area['y']
                        w = facial_area['w']
                        h = facial_area['h']
                        
                        embedding = np.array(rep['embedding'])
                        result = matcher.match(embedding)
                        
                        name = result['name']
                        confidence = result['confidence']
                        is_match = result['is_match']
                        
                        color = COLOR_KNOWN if is_match else COLOR_UNKNOWN
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
                        
                        label = f"{name} ({confidence*100:.0f}%)" if is_match else "Unknown"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                   FONT_SCALE, color, FONT_THICKNESS)
                
                processed_frames += 1
                if max_frames and processed_frames >= max_frames:
                    break
                    
            except Exception as e:
                pass  # Skip frame on error
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"\nVideo processing complete! Output saved to {output_path}")


if __name__ == "__main__":
    # Example usage in Colab:
    # 
    # # Upload an image
    # from google.colab import files
    # uploaded = files.upload()
    # image_path = list(uploaded.keys())[0]
    # 
    # # Recognize faces
    # recognize_faces_in_image(image_path)
    pass

