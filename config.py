"""
Configuration file for real-time face recognition system.

This file contains all configurable parameters including:
- Model selection (detector and embedder)
- Similarity threshold for matching
- Frame processing interval for performance
"""

# Face Detection Model
# Options: "retinaface" (preferred), "mtcnn"
DETECTOR = "retinaface"

# Face Recognition Model
# Options: "Facenet512" (preferred), "ArcFace", "VGG-Face"
MODEL_NAME = "Facenet512"

# Similarity Threshold
# Cosine similarity threshold for face matching
# Values closer to 1.0 = stricter matching (fewer false positives)
# Values closer to 0.0 = more lenient matching (more false positives)
THRESHOLD = 0.45

# Frame Processing Interval
# Process every Nth frame to balance speed and accuracy
# 1 = process every frame (slower but most accurate)
# 2 = process every other frame (faster, good balance)
FRAME_INTERVAL = 2

# Paths
DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "embeddings.pkl"

# Display Settings
WINDOW_NAME = "Real-Time Face Recognition"
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# Colors (BGR format for OpenCV)
COLOR_KNOWN = (0, 255, 0)  # Green for recognized faces
COLOR_UNKNOWN = (0, 0, 255)  # Red for unknown faces
COLOR_TEXT = (255, 255, 255)  # White text

# Camera/Stream Settings
RTSP_RETRY_ATTEMPTS = 5
RTSP_RETRY_DELAY = 5  # seconds

