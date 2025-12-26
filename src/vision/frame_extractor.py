import logging
import os
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

# Configure logger
logger = logging.getLogger("FashionEngine")


def extract_frames(
    video_path: str,
    target_fps: int = 1,
    max_frames: Optional[int] = None,
    resize_dim: Optional[Tuple[int, int]] = (512, 512),
) -> Generator[np.ndarray, None, None]:
    """
    Memory-efficient video frame extractor.

    Args:
        video_path: Path to the video file.
        target_fps: How many frames to extract per second of video.
        max_frames: Safety limit to stop processing (e.g., only scan first 100 frames).
        resize_dim: Tuple (width, height) to resize frames. None to keep original.

    Yields:
        np.ndarray: The image frame (BGR format).
    """

    if not os.path.exists(video_path):
        logger.error(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"❌ Could not open video file: {video_path}")
        return

    try:
        # Get metadata
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if native_fps == 0 or np.isnan(native_fps):
            logger.warning(
                f"⚠️ Video reports 0 FPS. Fallback to reading every 30th frame."
            )
            native_fps = 30.0

        # Calculate skip interval (e.g., if Native=30 and Target=2, skip 15 frames)
        skip_interval = max(int(native_fps // target_fps), 1)

        logger.info(
            f"📹 Processing '{os.path.basename(video_path)}': {native_fps:.2f} FPS | Step: {skip_interval}"
        )

        frame_count = 0
        yielded_count = 0

        while True:
            # Safety break
            if max_frames and yielded_count >= max_frames:
                break

            # Optimization: Use grab() for skipped frames (faster than read())
            # We only decode (retrieve) if it's the frame we want.
            if frame_count % skip_interval != 0:
                ret = cap.grab()  # Fast skip
                if not ret:
                    break
                frame_count += 1
                continue

            # Full decode for the frame we want
            ret, frame = cap.retrieve()
            if not ret:
                break

            # Resize if requested (Crucial for AI speed)
            if resize_dim:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)

            yield frame

            yielded_count += 1
            frame_count += 1

    except Exception as e:
        logger.error(f"🔥 Error extracting frames: {e}")

    finally:
        cap.release()
        logger.info(f"✅ Extraction complete. Processed {yielded_count} frames.")
