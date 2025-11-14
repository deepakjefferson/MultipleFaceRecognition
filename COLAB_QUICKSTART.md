# Quick Start: Google Colab

Copy and paste these cells into a new Colab notebook.

## Cell 1: Install Dependencies
```python
!pip install -q deepface opencv-python numpy tensorflow tf-keras
```

## Cell 2: Upload Project Files
```python
from google.colab import files
import zipfile
import os

# Upload your project zip file
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted {filename}")

# Create directories
os.makedirs('dataset', exist_ok=True)
os.makedirs('utils', exist_ok=True)
```

## Cell 3: Upload Dataset
```python
from google.colab import files
import zipfile

# Upload dataset zip
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Dataset extracted")
```

## Cell 4: Train
```python
!python train.py
```

## Cell 5: Recognize Faces in Image
```python
from recognize_colab import recognize_faces_in_image
from google.colab import files

# Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Recognize
recognize_faces_in_image(image_path)
```

## Cell 6: Download Results
```python
from google.colab import files
files.download('embeddings.pkl')
files.download('result.jpg')
```

## Tips:
- Enable GPU: Runtime → Change runtime type → GPU
- Save work frequently (sessions timeout)
- Download important files before session ends

