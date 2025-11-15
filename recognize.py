#!/usr/bin/env python3
"""
Real-time face recognition from webcam, IP camera, or RTSP stream.

This script:
1. Loads pre-trained embeddings from embeddings.pkl
2. Captures video from webcam, IP camera, or RTSP stream
3. Detects all faces in each frame
4. Recognizes each face by comparing embeddings
5. Draws bounding boxes with names
6. Displays live video feed

Usage:
    # Webcam
    python3 recognize.py --source 0
    
    # IP Camera (HTTP)
    python3 recognize.py --source "http://192.168.1.5:8080/video"
    
    # IP Camera with authentication (credentials in URL)
    python3 recognize.py --source "http://username:password@192.168.1.5:8080/video"
    
    # IP Camera with authentication (separate arguments)
    python3 recognize.py --source "http://192.168.1.5:8080/video" --username myuser --password mypass
    
    # RTSP Stream
    python3 recognize.py --source "rtsp://user:pass@192.168.1.5:554/stream"
"""

import argparse
import pickle
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse
import cv2
import numpy as np
from deepface import DeepFace
from config import (
    EMBEDDINGS_FILE,
    THRESHOLD,
    FRAME_INTERVAL,
    WINDOW_NAME,
    COLOR_KNOWN,
    COLOR_UNKNOWN,
    COLOR_TEXT,
    FONT_SCALE,
    FONT_THICKNESS,
    BOX_THICKNESS,
    RTSP_RETRY_ATTEMPTS,
    RTSP_RETRY_DELAY,
    DETECTOR,
    MODEL_NAME,
    PROCESSING_WIDTH,
    PROCESSING_HEIGHT
)
from utils.matcher import FaceMatcher


def load_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """
    Load embeddings from pickle file.
    
    Args:
        embeddings_path: Path to embeddings.pkl file
        
    Returns:
        Dictionary mapping person names to their embeddings
    """
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


def open_video_capture(source: str, username: Optional[str] = None, password: Optional[str] = None) -> Optional[cv2.VideoCapture]:
    """
    Open video capture from various sources.
    
    Supports:
    - Webcam: "0" or integer
    - IP Camera: "http://IP:port/video" or "http://user:pass@IP:port/video"
    - RTSP: "rtsp://user:pass@host:port/stream"
    
    Args:
        source: Video source string
        username: Optional username for HTTP authentication (if not in URL)
        password: Optional password for HTTP authentication (if not in URL)
        
    Returns:
        cv2.VideoCapture object or None if failed
    """
    # Check if source is a webcam index (integer)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        # Handle HTTP streams with authentication
        if source.startswith('http://') or source.startswith('https://'):
            # If credentials are provided separately and not in URL, add them
            if username and password and '@' not in source:
                # Parse the URL to insert credentials
                parsed = urlparse(source)
                # Reconstruct URL with credentials
                netloc = f"{username}:{password}@{parsed.hostname}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                new_parsed = parsed._replace(netloc=netloc)
                source = urlunparse(new_parsed)
        
        # IP camera, RTSP stream, or HTTP stream with credentials
        cap = cv2.VideoCapture(source)
    
    # Test if capture opened successfully
    if cap.isOpened():
        # Try to read a frame to verify connection
        ret, _ = cap.read()
        if ret:
            return cap
        else:
            cap.release()
            return None
    else:
        return None


def resize_for_processing(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Resize frame for faster processing while maintaining aspect ratio.
    
    Args:
        frame: Original frame
        
    Returns:
        Tuple of (resized_frame, scale_factor)
    """
    if PROCESSING_WIDTH is None or PROCESSING_HEIGHT is None:
        return frame, 1.0
    
    h, w = frame.shape[:2]
    
    # Calculate scale to fit within processing dimensions
    scale_w = PROCESSING_WIDTH / w
    scale_h = PROCESSING_HEIGHT / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, 1.0 / scale


def scale_bbox(bbox: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    """Scale bounding box coordinates back to original frame size."""
    x1, y1, x2, y2 = bbox
    return (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))


def draw_face_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    name: str,
    confidence: float,
    is_known: bool
):
    """
    Draw bounding box and label on frame.
    
    Args:
        frame: Image frame to draw on
        bbox: Bounding box (x1, y1, x2, y2)
        name: Person name or "Unknown"
        confidence: Similarity confidence (0.0 to 1.0)
        is_known: Whether this is a known person
    """
    x1, y1, x2, y2 = bbox
    
    # Choose color based on recognition status
    color = COLOR_KNOWN if is_known else COLOR_UNKNOWN
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # Prepare label text
    if is_known:
        label = f"{name} ({confidence*100:.0f}%)"
    else:
        label = "Unknown"
    
    # Calculate text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    
    # Draw text background rectangle
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 10),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        COLOR_TEXT,
        FONT_THICKNESS
    )


def main():
    """Main recognition function."""
    parser = argparse.ArgumentParser(description="Real-time face recognition")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Video source: webcam index (0), IP camera ("http://IP:port/video"), or RTSP ("rtsp://...")'
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=EMBEDDINGS_FILE,
        help=f"Path to embeddings file (default: {EMBEDDINGS_FILE})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Similarity threshold for matching (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username for HTTP/RTSP authentication (if not in URL)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password for HTTP/RTSP authentication (if not in URL)"
    )
    args = parser.parse_args()
    
    # Load embeddings
    embeddings_path = Path(args.embeddings)
    embeddings = load_embeddings(embeddings_path)
    
    if not embeddings:
        print("No embeddings loaded. Exiting.")
        sys.exit(1)
    
    # Initialize components
    print(f"\nInitializing face detector: {DETECTOR}")
    print(f"Initializing face embedder: {MODEL_NAME}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Frame processing interval: {FRAME_INTERVAL}")
    
    matcher = FaceMatcher(embeddings, threshold=args.threshold)
    
    # Open video capture
    source = args.source
    print(f"\nConnecting to video source: {source}")
    
    retry_count = 0
    cap = None
    
    while retry_count < RTSP_RETRY_ATTEMPTS:
        cap = open_video_capture(source, username=args.username, password=args.password)
        if cap is not None:
            print("Video stream connected successfully!")
            break
        else:
            retry_count += 1
            if retry_count < RTSP_RETRY_ATTEMPTS:
                print(f"Connection failed. Retrying in {RTSP_RETRY_DELAY} seconds... ({retry_count}/{RTSP_RETRY_ATTEMPTS})")
                time.sleep(RTSP_RETRY_DELAY)
            else:
                print("Failed to connect to video source after multiple attempts.")
                sys.exit(1)
    
    # Create display window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Frame counter for processing interval
    frame_count = 0
    
    # Results cache - store last recognition results to display on skipped frames
    last_results = []
    results_lock = threading.Lock()
    processing = False
    
    def process_frame_async(frame_to_process: np.ndarray, scale_factor: float):
        """Process frame in background thread."""
        nonlocal last_results, processing
        
        if processing:
            return  # Skip if already processing
        
        processing = True
        try:
            # Use DeepFace.represent() to get both detection and embedding in one call
            frame_rgb = frame_to_process[:, :, ::-1]  # BGR to RGB
            
            representations = DeepFace.represent(
                img_path=frame_rgb,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False,
                align=True
            )
            
            # DeepFace.represent() may return a list or a single dict
            if not representations:
                with results_lock:
                    last_results = []
                return
            
            # Ensure representations is a list
            if isinstance(representations, dict):
                representations = [representations]
            
            # Process each detected face
            new_results = []
            for rep in representations:
                if 'facial_area' in rep and 'embedding' in rep:
                    # Extract bounding box
                    facial_area = rep['facial_area']
                    x = facial_area['x']
                    y = facial_area['y']
                    w = facial_area['w']
                    h = facial_area['h']
                    bbox = (x, y, x + w, y + h)
                    
                    # Scale bbox back to original frame size
                    bbox = scale_bbox(bbox, scale_factor)
                    
                    # Get embedding
                    embedding = np.array(rep['embedding'])
                    
                    # Match against known faces
                    result = matcher.match(embedding)
                    
                    name = result['name']
                    confidence = result['confidence']
                    is_match = result['is_match']
                    
                    new_results.append({
                        'bbox': bbox,
                        'name': name,
                        'confidence': confidence,
                        'is_match': is_match
                    })
                    
                    # Print to terminal
                    if is_match:
                        print(f"Detected: {name}, Confidence: {confidence*100:.0f}%")
                    else:
                        print("Unknown person detected")
            
            # Update results cache
            with results_lock:
                last_results = new_results
                
        except Exception as e:
            # If face detection/recognition fails, clear results
            with results_lock:
                last_results = []
        finally:
            processing = False
    
    print("\nStarting face recognition...")
    print("Press 'q' or ESC to quit")
    print(f"Processing every {FRAME_INTERVAL} frames")
    if PROCESSING_WIDTH and PROCESSING_HEIGHT:
        print(f"Processing resolution: {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")
    print()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to read frame. Attempting to reconnect...")
                cap.release()
                time.sleep(RTSP_RETRY_DELAY)
                cap = open_video_capture(source, username=args.username, password=args.password)
                if cap is None:
                    print("Reconnection failed. Exiting.")
                    break
                continue
            
            frame_count += 1
            
            # Process every Nth frame (for performance)
            if frame_count % FRAME_INTERVAL == 0:
                # Resize frame for faster processing
                processing_frame, scale = resize_for_processing(frame.copy())
                
                # Process in background thread (non-blocking)
                thread = threading.Thread(
                    target=process_frame_async,
                    args=(processing_frame, scale),
                    daemon=True
                )
                thread.start()
            
            # Draw last known results on current frame (even if not processing this frame)
            with results_lock:
                for result in last_results:
                    draw_face_box(
                        frame,
                        result['bbox'],
                        result['name'],
                        result['confidence'],
                        result['is_match']
                    )
            
            # Display frame immediately (non-blocking)
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExiting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()

