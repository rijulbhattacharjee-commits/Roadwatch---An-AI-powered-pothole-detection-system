"""
Preprocessing and metadata utilities for RoadWatch.

Handles:
- File extension and size validation
- Image loading, RGB conversion, and resizing
- Video frame sampling for short road video clips
- Optional GPS coordinate extraction from image EXIF metadata
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import ExifTags, Image


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_MB = 50.0


def validate_file(
    filename: str,
    file_size_bytes: Optional[int] = None,
    allow_video: bool = True,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> Tuple[bool, str]:
    """
    Validate uploaded file extension and size.

    Returns:
        (is_valid, message)
    """
    if not filename:
        return False, "Filename is empty."

    ext = Path(filename).suffix.lower()
    allowed = set(ALLOWED_IMAGE_EXTENSIONS)
    if allow_video:
        allowed |= ALLOWED_VIDEO_EXTENSIONS

    if ext not in allowed:
        return (
            False,
            f"Unsupported file extension '{ext}'. Allowed: {sorted(allowed)}",
        )

    if file_size_bytes is not None:
        if file_size_bytes <= 0:
            return False, "File is empty (0 bytes)."
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            return (
                False,
                f"File size ({size_mb:.1f} MB) exceeds limit of {max_size_mb:.1f} MB.",
            )

    return True, "Valid file."


def load_and_preprocess_image(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    max_dim: Optional[int] = 1280,
) -> Tuple[np.ndarray, Optional[Image.Image]]:
    """
    Load an image from path, bytes, PIL Image, or numpy array into an RGB numpy array.
    Optionally downscales images whose largest dimension exceeds `max_dim` while
    preserving aspect ratio.

    Returns:
        (rgb_array, original_pil_image_or_None)
    """
    pil_img: Optional[Image.Image] = None

    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] == 4:
            rgb = cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        elif source.ndim == 3 and source.shape[2] == 3:
            rgb = source.copy()
        else:
            raise ValueError(f"Unsupported image array shape: {source.shape}")
    else:
        if isinstance(source, (str, Path)):
            pil_img = Image.open(source)
        elif isinstance(source, bytes):
            pil_img = Image.open(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            pil_img = Image.open(source)
        elif isinstance(source, Image.Image):
            pil_img = source
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        rgb = np.array(pil_img.convert("RGB"))

    if max_dim is not None and max_dim > 0:
        h, w = rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return rgb, pil_img


def _dms_to_decimal(dms: Sequence, ref: str) -> Optional[float]:
    """Convert EXIF GPS degrees/minutes/seconds tuple and reference ('N','S','E','W') to decimal degrees."""
    try:
        def _to_float(val) -> float:
            if isinstance(val, tuple) and len(val) == 2:
                return float(val[0]) / float(val[1]) if val[1] != 0 else 0.0
            return float(val)

        degrees = _to_float(dms[0])
        minutes = _to_float(dms[1])
        seconds = _to_float(dms[2])
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def extract_gps_exif(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image]
) -> Optional[Dict[str, float]]:
    """
    Extract GPS latitude and longitude from an image's EXIF metadata if present.

    Returns:
        {"latitude": float, "longitude": float} or None if no GPS metadata exists.
    """
    try:
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            img = Image.open(source)
        elif isinstance(source, Image.Image):
            img = source
        else:
            return None

        exif_data = img._getexif() if hasattr(img, "_getexif") else None
        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo" and isinstance(value, dict):
                for gps_id, gps_val in value.items():
                    gps_tag = ExifTags.GPSTAGS.get(gps_id, gps_id)
                    gps_info[gps_tag] = gps_val

        if not gps_info:
            return None

        lat_dms = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef", "N")
        lon_dms = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef", "E")

        if lat_dms and lon_dms:
            lat = _dms_to_decimal(lat_dms, str(lat_ref))
            lon = _dms_to_decimal(lon_dms, str(lon_ref))
            if lat is not None and lon is not None:
                return {"latitude": lat, "longitude": lon}
    except Exception:
        return None

    return None


def sample_video_frames(
    video_source: Union[str, Path, bytes],
    every_n_frames: int = 15,
    max_frames: int = 20,
    max_dim: int = 960,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Sample frames from a short video file for pothole analysis.

    Args:
        video_source: Path to video file or raw video bytes.
        every_n_frames: Sample 1 frame every N frames.
        max_frames: Maximum number of frames to extract.
        max_dim: Maximum width/height for resized frames.

    Returns:
        List of tuples: (frame_index, timestamp_seconds, rgb_frame_array)
    """
    temp_path: Optional[str] = None
    try:
        if isinstance(video_source, bytes):
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as f:
                f.write(video_source)
            video_path = temp_path
        else:
            video_path = str(video_source)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        sampled: List[Tuple[int, float, np.ndarray]] = []
        frame_idx = 0
        step = max(1, every_n_frames)

        while len(sampled) < max_frames:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if max_dim > 0:
                    h, w = rgb.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / float(max(h, w))
                        rgb = cv2.resize(
                            rgb,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                timestamp_sec = round(frame_idx / fps, 2)
                sampled.append((frame_idx, timestamp_sec, rgb))
            frame_idx += 1

        cap.release()
        return sampled
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
