# Real-Time Multi-Person Face Recognition

A complete Python project for real-time face recognition using deep learning models. This system detects all faces in video frames, recognizes them against a database of known faces, and displays results with bounding boxes and names.

## Features

- **Deep Learning Models**: Uses RetinaFace for detection and Facenet512/ArcFace for recognition (via DeepFace)
- **Multi-Person Recognition**: Detects and recognizes multiple faces simultaneously
- **Multiple Video Sources**: Supports webcam, IP cameras, and RTSP streams
- **Real-Time Processing**: Optimized for live video streams with configurable frame skipping
- **Easy Training**: Simple training process from organized image folders

## Requirements

- Python 3.12
- Webcam, IP camera, or RTSP stream
- GPU recommended (but not required)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd real_time_face_recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: DeepFace will automatically download required models on first use. This may take a few minutes.

3. **Prepare your dataset:**
   ```
   dataset/
   ├── PersonName1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── PersonName2/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

   Each subfolder in `dataset/` should be named after a person and contain multiple images of that person.

## Usage

### Step 1: Train the Model

Generate face embeddings from your dataset:

```bash
python3 train.py
```

This will:
- Process all images in `dataset/` subfolders
- Extract face embeddings for each person
- Average embeddings per person
- Save results to `embeddings.pkl`

**Options:**
```bash
python3 train.py --dataset /path/to/dataset --output embeddings.pkl
```

### Step 2: Run Recognition

Start real-time face recognition:

```bash
# Webcam (default camera)
python3 recognize.py --source 0

# IP Camera
python3 recognize.py --source "http://192.168.1.5:8080/video"

# RTSP Stream
python3 recognize.py --source "rtsp://user:pass@192.168.1.5:554/stream"
```

**Options:**
```bash
python3 recognize.py \
    --source 0 \
    --embeddings embeddings.pkl \
    --threshold 0.45
```

**Controls:**
- Press `q` or `ESC` to quit

## Configuration

Edit `config.py` to customize:

- **DETECTOR**: Face detection model (`"retinaface"` or `"mtcnn"`)
- **MODEL_NAME**: Recognition model (`"Facenet512"`, `"ArcFace"`, `"VGG-Face"`)
- **THRESHOLD**: Similarity threshold for matching (0.0 to 1.0, default: 0.45)
- **FRAME_INTERVAL**: Process every Nth frame (1 = every frame, 2 = every other frame)

## Output

### Terminal Output
```
Detected: Jeffy, Confidence: 92%
Detected: Anand, Confidence: 88%
Unknown person detected
```

### Visual Output
- **Green boxes**: Recognized faces with names and confidence
- **Red boxes**: Unknown faces labeled "Unknown"

## Project Structure

```
real_time_face_recognition/
│
├── dataset/              # Training images (one folder per person)
│   ├── Jeffy/
│   └── Anand/
│
├── embeddings.pkl        # Generated face embeddings (created by train.py)
│
├── config.py             # Configuration settings
├── train.py              # Training script
├── recognize.py          # Recognition script
│
├── utils/
│   ├── __init__.py
│   ├── detector.py       # Face detection using RetinaFace/MTCNN
│   ├── embedder.py       # Face embedding generation
│   └── matcher.py        # Cosine similarity matching
│
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

1. **Training (`train.py`)**:
   - Loads images from `dataset/PersonName/` folders
   - Detects faces using RetinaFace
   - Extracts embeddings using Facenet512
   - Averages embeddings per person
   - Saves to `embeddings.pkl`

2. **Recognition (`recognize.py`)**:
   - Loads embeddings from `embeddings.pkl`
   - Captures video frames from source
   - Detects all faces in each frame
   - Extracts embedding for each face
   - Compares with stored embeddings using cosine similarity
   - Draws bounding boxes and labels
   - Displays live video feed

3. **Matching Algorithm**:
   - Uses cosine similarity: `similarity = dot(A,B) / (norm(A) * norm(B))`
   - If similarity ≥ threshold → match found
   - If similarity < threshold → unknown person

## Performance Tips

- **Frame Interval**: Set `FRAME_INTERVAL = 2` to process every other frame (faster)
- **GPU**: DeepFace will use GPU if available (TensorFlow/CUDA)
- **Image Quality**: Use clear, front-facing images for training
- **Lighting**: Ensure good lighting conditions for both training and recognition

## Troubleshooting

**"No embeddings loaded" error:**
- Run `train.py` first to generate `embeddings.pkl`

**"Failed to connect to video source":**
- Check camera permissions
- Verify IP camera URL is correct
- For RTSP, check network connectivity and credentials

**Slow performance:**
- Increase `FRAME_INTERVAL` in `config.py`
- Reduce video resolution
- Use GPU if available

**Poor recognition accuracy:**
- Add more training images per person (10-20 recommended)
- Ensure training images show different angles/expressions
- Adjust `THRESHOLD` in `config.py` (lower = more lenient)

## License

This project is provided as-is for educational and personal use.

## Running in Google Colab

For detailed Colab instructions, see [COLAB_GUIDE.md](COLAB_GUIDE.md).

### Quick Colab Setup:

1. **Install dependencies:**
   ```python
   !pip install -q deepface opencv-python numpy tensorflow tf-keras
   ```

2. **Upload project files** to Colab (use Files sidebar or upload zip)

3. **Upload dataset** and train:
   ```python
   !python train.py
   ```

4. **Recognize faces in images:**
   ```python
   from recognize_colab import recognize_faces_in_image
   recognize_faces_in_image('uploaded_image.jpg')
   ```

**Note**: Colab doesn't support live webcam streaming. Use:
- Photo capture widget for single images
- Video file processing for videos
- See `recognize_colab.py` for Colab-optimized functions

## Notes

- First run will download model files (may take several minutes)
- Models are cached locally after first download
- Works best with front-facing faces in good lighting
- Supports multiple faces per frame simultaneously
- For Colab: Enable GPU (Runtime → Change runtime type → GPU) for faster processing

