#!/usr/bin/env python3
"""
Training script to generate facial embeddings from a dataset of labeled images.

Usage:
    python3 train.py --dataset /path/to/dataset --output /path/to/embeddings.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import DATASET_DIR, EMBEDDINGS_PATH
from utils.embedder import FaceEmbedder
from utils.helpers import (
    iter_image_paths,
    iter_person_directories,
    load_image,
    save_embeddings,
    timestamped_print,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate face embeddings from dataset images.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Path to the dataset directory (default: config.DATASET_DIR).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EMBEDDINGS_PATH,
        help="Output path for the embeddings pickle file (default: config.EMBEDDINGS_PATH).",
    )
    return parser.parse_args()


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute the mean embedding for a person."""
    stacked = np.vstack(embeddings)
    return np.mean(stacked, axis=0)


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset.resolve()
    output_path: Path = args.output.resolve()

    embedder = FaceEmbedder()
    all_embeddings: Dict[str, np.ndarray] = {}

    if not dataset_dir.exists():
        timestamped_print(f"Dataset directory {dataset_dir} does not exist.")
        return

    for person_name, person_dir in iter_person_directories(dataset_dir):
        images = list(iter_image_paths(person_dir))
        if not images:
            timestamped_print(f"Skipping {person_name}: no images found.")
            continue

        timestamped_print(f"Training {person_name}... {len(images)} images found, generating embeddings.")

        person_embeddings: List[np.ndarray] = []
        for image_path in images:
            frame = load_image(image_path)
            if frame is None:
                continue
            embedding, _ = embedder.extract_best_embedding(frame)
            if embedding is None:
                timestamped_print(f"  - No face detected in {image_path.name}, skipping.")
                continue
            person_embeddings.append(embedding)

        if not person_embeddings:
            timestamped_print(f"Warning: No usable embeddings for {person_name}, skipping.")
            continue

        mean_embedding = average_embeddings(person_embeddings)
        all_embeddings[person_name] = mean_embedding
        timestamped_print(f"  - {person_name}: {len(person_embeddings)} embeddings aggregated.")

    if not all_embeddings:
        timestamped_print("No embeddings generated. Ensure dataset contains detectable faces.")
        return

    save_embeddings(all_embeddings, output_path)
    timestamped_print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()


