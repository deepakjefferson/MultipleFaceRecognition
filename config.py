"""
Global configuration values for the face recognition project.

All paths are absolute to avoid ambiguity when scripts are launched from any
working directory. Update these values to match deployment requirements.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.pkl"

# Model settings
# `buffalo_l` bundles RetinaFace for detection and ArcFace for embeddings.
INSIGHTFACE_MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)  # RetinaFace input resolution

# Matching behaviour
SIMILARITY_THRESHOLD = 0.45  # Cosine similarity threshold for a positive match
FRAME_INTERVAL = 3  # Process every Nth frame to balance speed and accuracy
MIN_FACE_SCORE = 0.4  # Ignore detections below this confidence score

# Streaming and retry behaviour
RTSP_RETRY_ATTEMPTS = 5
RTSP_RETRY_DELAY_SECONDS = 5

# Display behaviour
WINDOW_NAME = "Face Recognition Debug"


