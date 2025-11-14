#!/usr/bin/env python3
"""
Training script to generate facial embeddings from a dataset of labeled images.

This script:
1. Loads images from dataset/ folder (each subfolder is a person's name)
2. Detects faces and extracts embeddings for each image
3. Averages embeddings per person
4. Saves embeddings to embeddings.pkl

Usage:
    python3 train.py
    python3 train.py --dataset /path/to/dataset --output embeddings.pkl
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2
from config import DATASET_DIR, EMBEDDINGS_FILE, DETECTOR, MODEL_NAME
from utils.detector import FaceDetector
from utils.embedder import FaceEmbedder


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format), or None if loading fails
    """
    try:
        image = cv2.imread(str(image_path))
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_image_paths(directory: Path) -> List[Path]:
    """
    Get all image file paths from a directory.
    
    Supported formats: .jpg, .jpeg, .png, .bmp
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    image_paths = []
    
    if not directory.exists():
        return image_paths
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix in image_extensions:
            image_paths.append(file_path)
    
    return image_paths


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute the average embedding from a list of embeddings.
    
    This creates a single representative embedding for a person by averaging
    all their face embeddings. This helps handle variations in lighting,
    angles, and expressions.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Average embedding vector
    """
    if not embeddings:
        return None
    
    # Stack all embeddings vertically and compute mean
    stacked = np.vstack(embeddings)
    return np.mean(stacked, axis=0)


def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: Path):
    """
    Save embeddings dictionary to a pickle file.
    
    Args:
        embeddings: Dictionary mapping person names to their embeddings
        output_path: Path to save the pickle file
    """
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved {len(embeddings)} person embeddings to {output_path}")


def load_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a pickle file.
    
    Args:
        embeddings_path: Path to the pickle file
        
    Returns:
        Dictionary mapping person names to their embeddings
    """
    try:
        with open(embeddings_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Generate face embeddings from dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_DIR,
        help=f"Path to dataset directory (default: {DATASET_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EMBEDDINGS_FILE,
        help=f"Output embeddings file (default: {EMBEDDINGS_FILE})"
    )
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_dir}' does not exist!")
        print(f"Please create the directory and add person folders with images.")
        return
    
    # Initialize detector and embedder
    print(f"Initializing face detector: {DETECTOR}")
    print(f"Initializing face embedder: {MODEL_NAME}")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    # Dictionary to store all person embeddings
    all_embeddings: Dict[str, np.ndarray] = {}
    
    # Process each person folder in the dataset
    person_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not person_folders:
        print(f"No person folders found in {dataset_dir}")
        print("Expected structure: dataset/PersonName/image1.jpg, image2.jpg, ...")
        return
    
    print(f"\nFound {len(person_folders)} person folders")
    print("=" * 50)
    
    for person_dir in person_folders:
        person_name = person_dir.name
        print(f"\nProcessing: {person_name}")
        
        # Get all images for this person
        image_paths = get_image_paths(person_dir)
        
        if not image_paths:
            print(f"  No images found in {person_name}, skipping...")
            continue
        
        print(f"  Found {len(image_paths)} images")
        
        # Extract embeddings for all images of this person
        person_embeddings: List[np.ndarray] = []
        
        for image_path in image_paths:
            # Load image
            frame = load_image(image_path)
            if frame is None:
                print(f"  Failed to load {image_path.name}, skipping...")
                continue
            
            # Detect face
            detections = detector.detect_faces(frame)
            
            if not detections:
                print(f"  No face detected in {image_path.name}, skipping...")
                continue
            
            # Extract embedding (use first detected face)
            # In training, we assume one face per image
            face_box = None
            if 'facial_area' in detections[0]:
                facial_area = detections[0]['facial_area']
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                face_box = (x, y, x + w, y + h)
            
            embedding = embedder.extract_embedding(frame, face_box)
            
            if embedding is not None:
                person_embeddings.append(embedding)
                print(f"  ✓ Extracted embedding from {image_path.name}")
            else:
                print(f"  ✗ Failed to extract embedding from {image_path.name}")
        
        # Average embeddings for this person
        if person_embeddings:
            mean_embedding = average_embeddings(person_embeddings)
            all_embeddings[person_name] = mean_embedding
            print(f"  ✓ {person_name}: Averaged {len(person_embeddings)} embeddings")
        else:
            print(f"  ✗ No valid embeddings for {person_name}, skipping...")
    
    # Save all embeddings
    if all_embeddings:
        save_embeddings(all_embeddings, output_path)
        print("\n" + "=" * 50)
        print(f"Training complete! Generated embeddings for {len(all_embeddings)} people:")
        for name in all_embeddings.keys():
            print(f"  - {name}")
    else:
        print("\nError: No embeddings were generated!")
        print("Please ensure:")
        print("  1. Dataset folder contains person subfolders")
        print("  2. Each subfolder contains images with detectable faces")
        print("  3. Images are in supported formats (.jpg, .png, etc.)")


if __name__ == "__main__":
    main()

