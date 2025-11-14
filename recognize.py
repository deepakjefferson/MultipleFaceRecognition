#!/usr/bin/env python3
"""
Real-time face recognition from webcam or RTSP stream.

Usage:
    python3 recognize.py --source 0
    python3 recognize.py --rtsp "rtsp://user:pass@host:port/stream" --no-display
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    EMBEDDINGS_PATH,
    FRAME_INTERVAL,
    RTSP_RETRY_ATTEMPTS,
    RTSP_RETRY_DELAY_SECONDS,
    SIMILARITY_THRESHOLD,
    WINDOW_NAME,
)
from utils.embedder import FaceEmbedder
from utils.helpers import load_embeddings, timestamped_print
from utils.matcher import FaceMatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize faces from webcam or CCTV feed.")
    parser.add_argument("--source", type=str, default="0", help="Camera index or path (default: 0).")
    parser.add_argument("--rtsp", type=str, help="RTSP URL for CCTV camera.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=EMBEDDINGS_PATH,
        help="Path to embeddings pickle file (default: config.EMBEDDINGS_PATH).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for recognition (default: config.SIMILARITY_THRESHOLD).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window for headless environments.",
    )
    return parser.parse_args()


class GracefulExit:
    """Context manager to handle SIGINT/SIGTERM cleanly."""

    def __init__(self) -> None:
        self._exit = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:  # pragma: no cover - system interaction
        timestamped_print(f"Received signal {signum}, shutting down.")
        self._exit = True

    @property
    def exit(self) -> bool:
        return self._exit


def open_capture(source: str) -> Optional[cv2.VideoCapture]:
    """Initialize a cv2.VideoCapture for the provided source string."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def draw_detection(frame: np.ndarray, name: str, detection_score: float, bbox: tuple[int, int, int, int]) -> None:
    """Draw bounding box and label on the frame."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{name} ({detection_score * 100:.1f}%)"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> None:
    args = parse_args()
    embeddings_path: Path = args.embeddings.resolve()
    embeddings = load_embeddings(embeddings_path)
    if not embeddings:
        timestamped_print("No embeddings loaded. Run train.py before recognize.py.")
        sys.exit(1)

    matcher = FaceMatcher(embeddings, threshold=args.threshold)
    embedder = FaceEmbedder()

    source = args.rtsp if args.rtsp else args.source

    # Operational guidance for installers / operators.
    timestamped_print(
        "Camera guidelines: use 1080p at 25-30 FPS, avoid backlighting, mount at ~5-6ft facing entry."
    )

    # Try to create display window unless explicitly disabled.
    display_enabled = not args.no_display
    if display_enabled:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error:
            timestamped_print("Display not available. Continuing in headless mode.")
            display_enabled = False

    exit_handler = GracefulExit()

    retry_attempt = 0
    capture: Optional[cv2.VideoCapture] = None
    frame_counter = 0

    while not exit_handler.exit:
        if capture is None or not capture.isOpened():
            if retry_attempt >= RTSP_RETRY_ATTEMPTS:
                timestamped_print("Maximum retry attempts reached. Exiting.")
                break
            timestamped_print(f"Connecting to video source ({source})... attempt {retry_attempt + 1}")
            capture = open_capture(source)
            if capture:
                timestamped_print("Video stream connected.")
                retry_attempt = 0
            else:
                retry_attempt += 1
                timestamped_print(f"Connection failed. Retrying in {RTSP_RETRY_DELAY_SECONDS} seconds.")
                time.sleep(RTSP_RETRY_DELAY_SECONDS)
                continue

        success, frame = capture.read()
        if not success or frame is None:
            timestamped_print("Frame grab failed. Re-initialising stream.")
            capture.release()
            capture = None
            retry_attempt += 1
            time.sleep(RTSP_RETRY_DELAY_SECONDS)
            continue

        frame_counter += 1
        process_frame = FRAME_INTERVAL <= 1 or frame_counter % FRAME_INTERVAL == 0

        if process_frame:
            embeddings_batch, detections = embedder.extract_embeddings(frame)
            if embeddings_batch.size == 0:
                timestamped_print("Unknown Person Detected")
            else:
                for embedding, detection in zip(embeddings_batch, detections):
                    result = matcher.match(embedding)
                    if result.is_match:
                        timestamped_print(
                            f"Person Detected: {result.name} | Confidence: {result.confidence_percent:.1f}%"
                        )
                        name = result.name
                    else:
                        timestamped_print("Unknown Person Detected")
                        name = "Unknown"

                    if display_enabled:
                        draw_detection(frame, name, detection.score, detection.rect)

        if display_enabled:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                timestamped_print("Exit requested by user input.")
                break

    if capture:
        capture.release()
    if display_enabled:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


