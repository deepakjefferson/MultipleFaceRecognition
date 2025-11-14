# Running Face Recognition in Google Colab

This guide will help you run the real-time face recognition project in Google Colab.

## Step 1: Upload Project Files to Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Create a new notebook**

3. **Upload project files** using one of these methods:

### Method A: Direct Upload (Recommended for small projects)
```python
# Run this in a Colab cell
from google.colab import files
import zipfile
import os

# Upload your project as a zip file
uploaded = files.upload()

# Extract if needed
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted {filename}")
```

### Method B: Use Google Drive
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive
!cp -r /content/drive/MyDrive/real_time_face_recognition /content/
```

### Method C: Clone from GitHub (if you push to GitHub)
```python
!git clone https://github.com/yourusername/real_time_face_recognition.git
```

## Step 2: Install Dependencies

Run this in a Colab cell:

```python
!pip install -q deepface opencv-python numpy tensorflow tf-keras
```

## Step 3: Prepare Dataset

### Option A: Upload dataset folder
```python
from google.colab import files
import zipfile

# Upload dataset zip file
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('dataset')
        print(f"Dataset extracted to dataset/")
```

### Option B: Upload individual images
```python
# Create person folders
import os
os.makedirs('dataset/PersonName', exist_ok=True)

# Upload images (you'll need to do this manually or use a loop)
from google.colab import files
uploaded = files.upload()

# Move files to dataset folder
import shutil
for filename in uploaded.keys():
    shutil.move(filename, f'dataset/PersonName/{filename}')
```

## Step 4: Train the Model

```python
# Run training
!python train.py
```

Or run directly in Python:
```python
exec(open('train.py').read())
```

## Step 5: Run Recognition

### For Webcam (Colab's built-in camera)

**Note**: Colab doesn't have direct webcam access like a local machine. Use one of these alternatives:

### Option A: Upload Video File
```python
# Upload a video file
from google.colab import files
uploaded = files.upload()

# Process video (modify recognize.py to process video files)
import cv2

video_path = list(uploaded.keys())[0]
cap = cv2.VideoCapture(video_path)

# Process frames...
```

### Option B: Use Colab's Camera Widget
```python
# Install ipywidgets
!pip install -q ipywidgets

from IPython.display import HTML, display
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
    async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = 'Capture Photo';
        div.appendChild(capture);
        
        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        
        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();
        
        // Resize the output to fit the video element.
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
        
        return new Promise((resolve) => {
            capture.onclick = async () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                stream.getVideoTracks()[0].stop();
                div.remove();
                resolve(await canvas.toDataURL('image/jpeg', quality));
            };
        });
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Take photo
photo_path = take_photo()

# Process photo with face recognition
from recognize import load_embeddings, FaceMatcher
from deepface import DeepFace
import cv2

embeddings = load_embeddings('embeddings.pkl')
matcher = FaceMatcher(embeddings)

frame = cv2.imread(photo_path)
frame_rgb = frame[:, :, ::-1]

representations = DeepFace.represent(
    img_path=frame_rgb,
    model_name='Facenet512',
    detector_backend='retinaface',
    enforce_detection=False,
    align=True
)

# Draw results
for rep in representations:
    if 'facial_area' in rep and 'embedding' in rep:
        facial_area = rep['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        bbox = (x, y, x + w, y + h)
        embedding = np.array(rep['embedding'])
        result = matcher.match(embedding)
        
        name = result['name']
        confidence = result['confidence']
        is_match = result['is_match']
        
        color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{name} ({confidence*100:.0f}%)" if is_match else "Unknown"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Display result
from IPython.display import Image, display
cv2.imwrite('result.jpg', frame)
display(Image('result.jpg'))
```

### Option C: Process Images from Drive
```python
# Process images stored in Google Drive
import cv2
from recognize import load_embeddings, FaceMatcher
from deepface import DeepFace
import numpy as np

embeddings = load_embeddings('embeddings.pkl')
matcher = FaceMatcher(embeddings)

# Load image
image_path = '/content/drive/MyDrive/test_image.jpg'
frame = cv2.imread(image_path)

# Process (same as Option B above)
# ...
```

## Step 6: Download Results

```python
from google.colab import files

# Download embeddings
files.download('embeddings.pkl')

# Download result images
files.download('result.jpg')
```

## Complete Colab Notebook Example

Here's a complete notebook you can use:

```python
# Cell 1: Install dependencies
!pip install -q deepface opencv-python numpy tensorflow tf-keras

# Cell 2: Setup project structure
import os
os.makedirs('dataset', exist_ok=True)
os.makedirs('utils', exist_ok=True)

# Upload your project files here (config.py, train.py, recognize.py, utils/*.py)

# Cell 3: Upload dataset
from google.colab import files
import zipfile

uploaded = files.upload()
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted {filename}")

# Cell 4: Train
!python train.py

# Cell 5: Test on uploaded image
from google.colab import files
uploaded = files.upload()

# Process uploaded image (use code from Option B above)
```

## Important Notes for Colab

1. **Session Timeout**: Colab sessions timeout after inactivity. Save your work frequently.

2. **GPU Access**: 
   - Go to Runtime → Change runtime type → Select GPU
   - This speeds up DeepFace processing significantly

3. **File Persistence**: 
   - Files are deleted when session ends
   - Save important files to Google Drive or download them

4. **Webcam Limitations**: 
   - Colab doesn't support live webcam streaming like local Python
   - Use photo capture widget or process uploaded videos/images

5. **Memory**: 
   - Colab has limited RAM
   - Process videos frame-by-frame for large files

## Quick Start Template

Copy this into a Colab notebook:

```python
# Setup
!pip install -q deepface opencv-python numpy tensorflow tf-keras

# Upload project files manually or via Drive
# Then run:
!python train.py

# For recognition, use the photo capture widget or process uploaded images
```

## Troubleshooting

- **Import errors**: Make sure all project files are uploaded
- **Model download**: First run will download models (may take time)
- **GPU errors**: Try Runtime → Restart runtime
- **Memory errors**: Process smaller batches or use CPU

